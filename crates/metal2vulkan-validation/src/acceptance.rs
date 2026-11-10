//! Driver acceptance for the engine.
//!
//! `corpus-pipeline --profile P` compiles each row through the public compile API
//! and asks a Vulkan driver to create the pipeline **from the binder layout the artifact
//! carries** ([`PipelineLayout`](metal2vulkan::pipeline::PipelineLayout), vertex input, stage
//! modules with their specialization and subgroup facts, the graphics state). Nothing is bound or
//! dispatched. A row is built as [`PipelineRequest::Row`] says ([`row`]): a kernel's compute
//! pipeline; a vertex function's rasterization-disabled pipeline with a vertex descriptor for the
//! attributes it declares; a fragment function's rasterizing pipeline drawn with its generated
//! passthrough vertex companion ([`crate::companion`], the one the executors draw with).
//! This module is the device-free half: compiling a row and deciding whether the opened device
//! enabled and provides what the pipeline requires ([`unenabled`] for the API version, features,
//! and extensions, by the product's `DeviceDescription::required_enables` on the opened
//! description; [`unmet_device_facts`] for properties, limits, formats, stage create flags, and
//! float-controls independence, against the opened device's queried profile). The device half is
//! [`crate::candidate::create_acceptance_pipeline`].
//!
//! The sweep compiles every row in a killable worker ([`compile_row_in_worker`]), never in the
//! process that holds the device; [`compile_pipeline`] is the compile the worker runs, and it
//! derives a row's descriptor and companion from the AIR there too.

pub mod dispatch;
pub mod metal_report;
pub mod row;

use crate::candidate::engine::graphics::RenderTemplate;
use crate::candidate::engine::mesh::MeshTemplate;
use crate::engine::{bounded_detail, FailureKind, Outcome, RowSource};
use crate::executor_contract::stage_inputs;
use crate::worker::compile::{
    compile_compute_request, compile_render_request, CompileWorker, NotCompiled, WorkerDevice,
};
use crate::worker::{Measured, WorkerFailure};
use dispatch::{BinderAnswer, ComputeCreation, NoCreation};
use metal2vulkan::bind::{BindShape, Dispatch};
use metal2vulkan::cache::VariantKey;
use metal2vulkan::device::Independence;
use metal2vulkan::device::Requirements;
use metal2vulkan::device::{Device, DeviceDescription, DevicePolicy, DeviceProfile, EnableSet};
use metal2vulkan::error::{DescriptorError, Error, RuntimeInput};
use metal2vulkan::library::{FunctionConstantValues, Library};
use metal2vulkan::metal::PixelFormat;
use metal2vulkan::pipeline::Stage;
use metal2vulkan::pipeline::{
    ComputePipeline, ComputePipelineDescriptor, FunctionDescriptor, FunctionHandle, LinkedFunction,
    LinkedFunctions, PrecisionNote, RenderPipeline, RenderPipelineDescriptor, StageInputDescriptor,
    StageModule, StageSource,
};
use metal2vulkan::vk;
use metal2vulkan::Budget;
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::collections::BTreeMap;
use std::ops::Range;
use std::time::{Duration, Instant};

/// A compiled pipeline: what a [`PipelineRequest`] builds. An L4 row's is made ready for a driver
/// as a [`RowPipeline`]. Each is boxed: the two kinds differ in size by hundreds of bytes.
#[derive(Clone, Debug)]
pub enum AcceptancePipeline {
    Compute(Box<ComputePipeline>),
    /// A render pipeline: a vertex function alone with rasterization disabled, or a vertex and a
    /// fragment function drawn together ([`PipelineRequest::Draw`]).
    Render(Box<RenderPipeline>),
}

impl AcceptancePipeline {
    /// What the pipeline needs from the device.
    pub fn requirements(&self) -> &Requirements {
        match self {
            AcceptancePipeline::Compute(p) => &p.requirements,
            AcceptancePipeline::Render(p) => &p.requirements,
        }
    }

    /// Total SPIR-V words over the pipeline's stages.
    pub fn words(&self) -> u64 {
        match self {
            AcceptancePipeline::Compute(p) => p.stage.spirv.len() as u64,
            AcceptancePipeline::Render(p) => p.stages.iter().map(|s| s.spirv.len() as u64).sum(),
        }
    }

    /// Every precision note of the pipeline's stages, in stage order, each once.
    pub fn precision_notes(&self) -> Vec<PrecisionNote> {
        match self {
            AcceptancePipeline::Compute(p) => stage_precision_notes(std::slice::from_ref(&p.stage)),
            AcceptancePipeline::Render(p) => stage_precision_notes(&p.stages),
        }
    }
}

/// Every precision note of `stages`, in stage order, each once.
fn stage_precision_notes(stages: &[StageModule]) -> Vec<PrecisionNote> {
    let mut notes = Vec::new();
    for note in stages.iter().flat_map(|stage| &stage.precision_notes) {
        if !notes.contains(note) {
            notes.push(*note);
        }
    }
    notes
}

/// The table handle of each function one request function links: by library of
/// [`RequestFunction::linked`], then by function of that library's [`LinkedLibrary::functions`],
/// what `LinkedFunctions::function_handle` answers for it over the linked functions the compile
/// built (`None` where the product gives it none). A visible-function table of the function's stage
/// holds these words.
pub type LinkedHandles = Vec<Vec<Option<FunctionHandle>>>;

/// What [`compile_pipeline`] built: the pipeline and, for each of the request's functions
/// ([`PipelineRequest::functions`]) in order, the handles of the functions it links.
#[derive(Clone, Debug)]
pub struct CompiledRequest {
    pub pipeline: AcceptancePipeline,
    pub handles: Vec<LinkedHandles>,
}

/// An L4 row compiled for a driver: a kernel with the dispatch region its pipeline is created from
/// ([`dispatch::plan_compute_row`]), or a render pipeline, created at its stages' default
/// specializations, of a `row` function.
#[derive(Clone, Debug)]
pub enum RowPipeline {
    Compute {
        pipeline: ComputePipeline,
        creation: ComputeCreation,
    },
    Render {
        pipeline: RenderPipeline,
        row: RowStage,
    },
}

impl RowPipeline {
    /// The row's pipeline as compiled: compute rows from [`compile_row`] have a creation region,
    /// render rows the stage of their own function.
    fn of(
        device: &Device,
        pipeline: AcceptancePipeline,
        stage: RowStage,
    ) -> Result<RowPipeline, NoCreation> {
        Ok(match pipeline {
            AcceptancePipeline::Compute(pipeline) => {
                let creation = dispatch::plan_compute_row(device, &pipeline)?;
                RowPipeline::Compute {
                    pipeline: *pipeline,
                    creation,
                }
            }
            AcceptancePipeline::Render(pipeline) => RowPipeline::Render {
                pipeline: *pipeline,
                row: stage,
            },
        })
    }

    /// What the pipeline needs from the device.
    pub fn requirements(&self) -> &Requirements {
        match self {
            RowPipeline::Compute { pipeline, .. } => &pipeline.requirements,
            RowPipeline::Render { pipeline, .. } => &pipeline.requirements,
        }
    }

    /// The pipeline's stages.
    pub fn stages(&self) -> &[StageModule] {
        match self {
            RowPipeline::Compute { pipeline, .. } => std::slice::from_ref(&pipeline.stage),
            RowPipeline::Render { pipeline, .. } => &pipeline.stages,
        }
    }

    /// Every precision note of the pipeline's stages, the row's own and its companions', in stage
    /// order, each once.
    pub fn precision_notes(&self) -> Vec<PrecisionNote> {
        stage_precision_notes(self.stages())
    }

    /// The Vulkan stage the row's own function compiled to: a kernel's compute stage, a fragment
    /// function's fragment stage, a vertex function's vertex stage, or a post-tessellation
    /// vertex function's evaluation stage.
    pub fn own_stage(&self) -> vk::ShaderStage {
        match self {
            RowPipeline::Compute { .. } => vk::ShaderStage::COMPUTE,
            RowPipeline::Render {
                row: RowStage::Fragment,
                ..
            } => vk::ShaderStage::FRAGMENT,
            RowPipeline::Render { pipeline, .. } if pipeline.graphics.tessellation.is_some() => {
                vk::ShaderStage::TESSELLATION_EVALUATION
            }
            RowPipeline::Render { .. } => vk::ShaderStage::VERTEX,
        }
    }

    /// The SPIR-V words of the row's own function's stage ([`RowPipeline::own_stage`]): the
    /// module the translation sweep compiles for the row.
    pub fn words(&self) -> u64 {
        let own = self.own_stage();
        self.stage_words(|stage| stage == own)
    }

    /// The SPIR-V words of every other stage: a fragment row's generated passthrough vertex
    /// companion, a post-tessellation vertex row's synthesized vertex and control stages; 0 for a
    /// kernel or a vertex function alone.
    pub fn companion_words(&self) -> u64 {
        let own = self.own_stage();
        self.stage_words(|stage| stage != own)
    }

    fn stage_words(&self, of: impl Fn(vk::ShaderStage) -> bool) -> u64 {
        self.stages()
            .iter()
            .filter(|stage| of(stage.stage))
            .map(|stage| stage.spirv.len() as u64)
            .sum()
    }
}

/// What compiling one row produced.
#[derive(Debug)]
pub enum RowCompile {
    /// The row's pipeline, ready for a driver.
    Compiled(Box<RowPipeline>),
    /// The compile refused or failed: the typed outcome and a bounded detail.
    NotCompiled(Outcome, String),
    /// The tool does not build this row's pipeline inputs ([`PipelineCompileError::Unmodelled`]):
    /// a vertex attribute of a type no Metal vertex format reads, or a fragment function no
    /// generated companion draws, by kind. A modelled limit of the tool, never a product verdict.
    Unmodelled(row::Unmodelled),
    /// The binder's error for a kernel row's representative dispatch ([`dispatch`]): a typed
    /// refusal the product documents, or an error that fails the sweep.
    Binder(BinderAnswer),
    /// The worker boundary stopped the compile or its result was unusable (a deadline, the memory
    /// limit, a crashed worker, an artifact that does not load): the outcome
    /// ([`crate::worker::WorkerFailure::outcome`]) and the failure text. A budget breach is a
    /// failing result (AGENTS.md), so it fails the sweep.
    Stopped(Outcome, String),
}

impl RowCompile {
    /// A compiled row, or why the tool creates no pipeline from it.
    fn of(pipeline: Result<RowPipeline, NoCreation>) -> RowCompile {
        match pipeline {
            Ok(pipeline) => RowCompile::Compiled(Box::new(pipeline)),
            Err(NoCreation::Unmodelled(why)) => RowCompile::Unmodelled(why),
            Err(NoCreation::Binder(answer)) => RowCompile::Binder(answer),
        }
    }
}

/// How an L4 row's vertex function fetches its attributes ([`row`]): the projection the
/// product derives from the synthesized descriptor (fixed-function wherever the device fetches
/// the format), or every vertex buffer pulled in the shader (`VariantKey::pulled_vertex_buffers`),
/// the repeatable form of the pulled sweep.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Projection {
    #[default]
    Derived,
    Pulled,
}

/// The pipeline one compile builds: its kind, its functions, their function-constant values, and
/// the functions each links. Its functions come from the compile's libraries
/// ([`PipelineRequest::libraries`]): a draw's vertex function from the first and its fragment
/// function from the second, a mesh draw's each from the one its [`MeshFunction::library`] names,
/// every other request's function from the one, and then the libraries of each function's linked
/// functions ([`RequestFunction::linked`]).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PipelineRequest {
    /// `Device::compile_compute_pipeline` of kernel `entry` with `constants`.
    Compute {
        entry: String,
        constants: FunctionConstantValues,
    },
    /// Kernel `function`, compiled with the lowering inputs the bind shape `shape` decides
    /// (`candidate::engine::variants`), and those `dispatch` runs on when given (a partial
    /// simdgroup's emulated simdgroups): the engine candidate's dispatch.
    BoundCompute {
        function: RequestFunction,
        shape: BindShape,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        dispatch: Option<Dispatch>,
        /// The compute pipeline's stage-input descriptor, which the kernel compiles under.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        stage_input: Option<StageInputDescriptor>,
        /// A tile function's color attachment formats by color index: the kernel compiles as
        /// a tile function under them, into `Device::compile_tile_pipeline`'s pipeline
        /// (`variants::compile_tile_for_shape`). A tile pipeline has no stage-input descriptor.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        tile: Option<BTreeMap<u32, PixelFormat>>,
        /// The kernel compiles as a device-replay variant (`LoweringInputs::replay_bindings`,
        /// `variants::compile_replay_for_shape`): a replayed template's pipeline, its buffers read
        /// through the slot table the replay's preprocess writes per command. A replay variant has
        /// no stage-input descriptor and no tile pass.
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        replay: bool,
    },
    /// The L4 row `entry`, a `stage` function with the function-constant values `constants`
    /// (none, the default: Metal's undefined, the plain census's compile), built as [`row`]
    /// derives it from the library's AIR under those values, its vertex attributes fetched under
    /// `projection`.
    Row {
        entry: String,
        stage: RowStage,
        projection: Projection,
        #[serde(default)]
        constants: FunctionConstantValues,
    },
    /// A rasterization-disabled `Device::compile_render_pipeline` of vertex function `function`,
    /// from the Metal oracle's descriptor `template` (its vertex descriptor, color attachments,
    /// and topology class): the engine candidate's vertex draw. With a bind `shape`, the stages
    /// are compiled with the inputs that shape decides (`variants::compile_render_for_shape`:
    /// storage-texture formats, pulled vertex buffers).
    VertexDraw {
        function: RequestFunction,
        template: RenderTemplate,
        #[serde(default)]
        shape: Option<BindShape>,
        /// With a `shape`, the form of the variant it decides the stages compile in
        /// (`variants::DrawForm`): the plain one, or an executed render command's, the
        /// large-buffer form the runtime's host step draws in or a draw replay's forms its GPU
        /// form draws in.
        #[serde(
            default,
            skip_serializing_if = "crate::candidate::engine::variants::DrawForm::is_bound"
        )]
        form: crate::candidate::engine::variants::DrawForm,
    },
    /// `Device::compile_render_pipeline` of `vertex` and `fragment`, rasterization enabled, from
    /// the Metal oracle's descriptor `template`: the engine candidate's draw of a case and its
    /// generated companion, each from its own library (a fragment case's passthrough vertex, an
    /// observed vertex case's observer fragment), with the inputs a bind `shape` decides as for
    /// [`PipelineRequest::VertexDraw`].
    Draw {
        vertex: RequestFunction,
        fragment: RequestFunction,
        template: RenderTemplate,
        #[serde(default)]
        shape: Option<BindShape>,
    },
    /// `Device::compile_mesh_pipeline` of a mesh render pipeline from the Metal oracle's
    /// descriptor `template`, of its object function where it has one, its mesh function, and its
    /// fragment function where it has one, each from the stage library its
    /// [`MeshFunction::library`] names (the case's, then its generated companion's): the engine
    /// candidate's mesh draw (`engine::mesh::mesh_request`). With a bind `shape`, the pipeline is
    /// the variant that shape decides (`variants::compile_mesh_for_shape`), as the runtime
    /// compiles a mesh draw's.
    MeshDraw {
        object: Option<MeshFunction>,
        mesh: MeshFunction,
        fragment: Option<MeshFunction>,
        template: MeshTemplate,
        #[serde(default)]
        shape: Option<BindShape>,
    },
}

/// One stage's function of a [`PipelineRequest::MeshDraw`] and the stage library it comes from,
/// by index among the request's stage libraries ([`PipelineRequest::stage_libraries`]).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MeshFunction {
    pub library: usize,
    pub function: RequestFunction,
}

/// One function of a request: its entry, its function-constant values, and the functions it
/// links (`MTLLinkedFunctions`).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RequestFunction {
    pub entry: String,
    pub constants: FunctionConstantValues,
    /// The libraries this function links functions from, in the compile's library order: after
    /// the request's stage libraries and the libraries every earlier function links, in stage
    /// order ([`PipelineRequest::libraries`]).
    pub linked: Vec<LinkedLibrary>,
}

impl RequestFunction {
    /// `entry` with `constants`, linking nothing.
    pub fn new(entry: impl Into<String>, constants: FunctionConstantValues) -> RequestFunction {
        RequestFunction {
            entry: entry.into(),
            constants,
            linked: Vec::new(),
        }
    }
}

/// One library a request function links functions from: the Metal name of each function linked from
/// it, into the product's `LinkedFunctions.functions`, as the Metal oracle puts every linked
/// function into `MTLLinkedFunctions.functions`. Each is specialized as Metal's
/// `newFunctionWithName:constantValues:` specializes it with the request function's constants:
/// with the values of the constants its library declares, the others ignored.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LinkedLibrary {
    pub functions: Vec<String>,
}

impl PipelineRequest {
    /// How many libraries the request's stage functions come from: two for a draw (the vertex
    /// function's, then the fragment function's), as many as a mesh draw's functions name, one
    /// otherwise (an L4 fragment row's companion is derived from its one library, [`row`]).
    pub fn stage_libraries(&self) -> usize {
        match self {
            PipelineRequest::Draw { .. } => 2,
            PipelineRequest::MeshDraw {
                object,
                mesh,
                fragment,
                ..
            } => object
                .iter()
                .chain([mesh])
                .chain(fragment)
                .map(|f| f.library + 1)
                .max()
                .unwrap_or(1),
            PipelineRequest::Compute { .. }
            | PipelineRequest::BoundCompute { .. }
            | PipelineRequest::Row { .. }
            | PipelineRequest::VertexDraw { .. } => 1,
        }
    }

    /// The request's functions that can link functions, in stage order.
    pub fn functions(&self) -> Vec<&RequestFunction> {
        match self {
            PipelineRequest::Compute { .. } | PipelineRequest::Row { .. } => Vec::new(),
            PipelineRequest::BoundCompute { function, .. }
            | PipelineRequest::VertexDraw { function, .. } => vec![function],
            PipelineRequest::Draw {
                vertex, fragment, ..
            } => vec![vertex, fragment],
            PipelineRequest::MeshDraw {
                object,
                mesh,
                fragment,
                ..
            } => object
                .iter()
                .chain([mesh])
                .chain(fragment)
                .map(|f| &f.function)
                .collect(),
        }
    }

    /// How many libraries the compile takes, in this order: the stage libraries
    /// ([`PipelineRequest::stage_libraries`]), then each function's linked libraries
    /// ([`RequestFunction::linked`]), function by function in stage order.
    pub fn libraries(&self) -> usize {
        self.stage_libraries()
            + self
                .functions()
                .iter()
                .map(|function| function.linked.len())
                .sum::<usize>()
    }

    /// The compile's libraries each of [`PipelineRequest::functions`] links, by index into the
    /// libraries [`PipelineRequest::libraries`] counts: the first function's right after the
    /// stage libraries, each next function's right after the previous one's.
    pub fn linked_ranges(&self) -> Vec<Range<usize>> {
        let mut next = self.stage_libraries();
        self.functions()
            .iter()
            .map(|function| {
                let start = next;
                next += function.linked.len();
                start..next
            })
            .collect()
    }

    /// Whether `handles` answer this request ([`CompiledRequest::handles`]): one entry per
    /// function, one list per linked library, one handle per function linked from it.
    pub fn check_handles(&self, handles: &[LinkedHandles]) -> Result<(), String> {
        let functions = self.functions();
        let shape = |linked: &[LinkedLibrary]| {
            linked
                .iter()
                .map(|library| library.functions.len())
                .collect::<Vec<_>>()
        };
        let answered = |handles: &LinkedHandles| handles.iter().map(Vec::len).collect::<Vec<_>>();
        if functions.len() == handles.len()
            && functions
                .iter()
                .zip(handles)
                .all(|(function, handles)| shape(&function.linked) == answered(handles))
        {
            Ok(())
        } else {
            Err(format!(
                "the handles {:?} do not answer the linked functions {:?}",
                handles.iter().map(answered).collect::<Vec<_>>(),
                functions
                    .iter()
                    .map(|function| shape(&function.linked))
                    .collect::<Vec<_>>()
            ))
        }
    }
}

/// Why [`compile_pipeline`] built no pipeline.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PipelineCompileError {
    /// The product's typed refusal or failure.
    Product(Error),
    /// The compile was given another number of libraries than its request takes
    /// ([`PipelineRequest::libraries`]): the tool's exchange error, never a product outcome. The
    /// compile worker reports it as a refused input.
    Libraries { takes: usize, given: usize },
    /// The tool does not build the request's pipeline inputs for this function (an L4 row's
    /// descriptor or companion, [`row`]): a modelled limit of the tool.
    Unmodelled(row::Unmodelled),
    /// The host's tooling failed while building the request's inputs (`llvm-as` assembling an L4
    /// fragment row's generated companion), or the request names inputs no pipeline takes
    /// together (a tile pipeline's attachments with a stage-input descriptor): neither a product
    /// outcome nor a modelled limit.
    Tooling(String),
}

impl From<Error> for PipelineCompileError {
    fn from(error: Error) -> PipelineCompileError {
        PipelineCompileError::Product(error)
    }
}

impl std::fmt::Display for PipelineCompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PipelineCompileError::Product(error) => error.fmt(f),
            PipelineCompileError::Unmodelled(why) => why.fmt(f),
            PipelineCompileError::Tooling(why) => f.write_str(why),
            PipelineCompileError::Libraries { takes, given } => write!(
                f,
                "the compile request takes {takes} libraries, and the compile has {given}"
            ),
        }
    }
}

/// The pipeline stage an indexed row's `stage` names.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RowStage {
    Kernel,
    Vertex,
    Fragment,
    /// An object function.
    Object,
    /// A mesh function.
    Mesh,
}

impl RowStage {
    /// Parse an index `stage` value, a stage label of the one vocabulary
    /// ([`crate::case::Stage::metadata_label`]: `Kernel`, `Vertex`, `Fragment`, `Object`, `Mesh`).
    pub fn parse(stage: &str) -> Result<RowStage, String> {
        use crate::case::Stage;
        Ok(
            match Stage::from_metadata_label(stage)
                .ok_or_else(|| format!("unknown stage {stage:?}"))?
            {
                Stage::Kernel => RowStage::Kernel,
                Stage::Vertex => RowStage::Vertex,
                Stage::Fragment => RowStage::Fragment,
                Stage::Object => RowStage::Object,
                Stage::Mesh => RowStage::Mesh,
            },
        )
    }
}

/// Compile the L4 row `entry` (a `stage` function) of `bitcode` for `device` with no function
/// constants, under the standard budget with `deadline`, in this process ([`compile_pipeline`] of
/// [`PipelineRequest::Row`]). Every public entry point runs under `catch_unwind`, so a panic is
/// `NotCompiled(failed panic)`, never an unwind.
pub fn compile_row(
    bitcode: &[u8],
    entry: &str,
    stage: RowStage,
    projection: Projection,
    device: &Device,
    deadline: Option<Instant>,
) -> RowCompile {
    let request = PipelineRequest::Row {
        entry: entry.into(),
        stage,
        projection,
        constants: FunctionConstantValues::defaults(),
    };
    match compile_pipeline(&[bitcode], &request, device, deadline) {
        Ok(compiled) => RowCompile::of(RowPipeline::of(device, compiled.pipeline, stage)),
        Err(PipelineCompileError::Product(error)) => {
            RowCompile::NotCompiled(Outcome::from(&error), bounded_detail(&error.to_string()))
        }
        Err(PipelineCompileError::Unmodelled(why)) => RowCompile::Unmodelled(why),
        Err(
            error @ (PipelineCompileError::Libraries { .. } | PipelineCompileError::Tooling(_)),
        ) => RowCompile::NotCompiled(
            Outcome::Failed {
                failure: FailureKind::Tooling,
            },
            bounded_detail(&error.to_string()),
        ),
    }
}

/// [`compile_row`] of an indexed row's `source` (its retained bitcode, or its AIR text, which the
/// worker assembles) with the function-constant values `constants` in a killable compile worker
/// for `device`: the 20-s deadline, the 500-MiB process-group limit, and the group kill apply to
/// assembling, deriving, and compiling. Returns what the parent measured around the worker, when
/// one ran.
pub fn compile_row_in_worker(
    worker: &CompileWorker,
    device: &WorkerDevice,
    source: &RowSource<'_>,
    entry: &str,
    stage: RowStage,
    projection: Projection,
    constants: &FunctionConstantValues,
) -> (RowCompile, Option<Measured>) {
    let request = PipelineRequest::Row {
        entry: entry.into(),
        stage,
        projection,
        constants: constants.clone(),
    };
    let compiled = match stage {
        RowStage::Kernel => {
            compile_compute_request(worker, device, std::slice::from_ref(source), &request).map(
                |compiled| {
                    (
                        compiled.result.map(|compiled| {
                            AcceptancePipeline::Compute(Box::new(compiled.pipeline))
                        }),
                        compiled.measured,
                    )
                },
            )
        }
        RowStage::Object | RowStage::Mesh => {
            return (RowCompile::Unmodelled(row::mesh_pipeline(entry)), None)
        }
        RowStage::Vertex | RowStage::Fragment => {
            compile_render_request(worker, device, std::slice::from_ref(source), &request).map(
                |compiled| {
                    (
                        compiled.result.map(|compiled| {
                            AcceptancePipeline::Render(Box::new(compiled.pipeline))
                        }),
                        compiled.measured,
                    )
                },
            )
        }
    };
    worker_row(device, compiled, stage)
}

/// What a compile worker answered for a row whose own function is a `stage` function: the
/// pipeline made ready for a driver, the compile's refusal, or the boundary's stop, with what the
/// parent measured around the worker when one ran.
fn worker_row(
    device: &WorkerDevice,
    compiled: Result<(Result<AcceptancePipeline, NotCompiled>, Measured), WorkerFailure>,
    stage: RowStage,
) -> (RowCompile, Option<Measured>) {
    match compiled {
        Ok((result, measured)) => (
            match result {
                Ok(pipeline) => RowCompile::of(RowPipeline::of(device.device(), pipeline, stage)),
                Err(NotCompiled::Product(outcome, detail)) => {
                    RowCompile::NotCompiled(outcome, detail)
                }
                Err(NotCompiled::Unmodelled(why)) => RowCompile::Unmodelled(why),
            },
            Some(measured),
        ),
        Err(failure) => (
            RowCompile::Stopped(failure.outcome(), bounded_detail(&failure.message())),
            failure.measured(),
        ),
    }
}

/// A logged draw (`corpus-pipeline --draws`): vertex function `vertex` of the first of
/// `libraries` and fragment function `fragment` of the second, each an indexed row's source,
/// compiled together under `template`, the descriptor the application created the pipeline with,
/// in a killable compile worker for `device` as [`compile_row_in_worker`] compiles a row (the
/// same deadline, memory limit and group kill), with no function-constant values. The pipeline
/// comes back as a fragment row's: its own stage is the fragment stage.
pub fn compile_draw_in_worker(
    worker: &CompileWorker,
    device: &WorkerDevice,
    libraries: &[RowSource<'_>; 2],
    vertex: &str,
    fragment: &str,
    template: &RenderTemplate,
) -> (RowCompile, Option<Measured>) {
    let request = PipelineRequest::Draw {
        vertex: RequestFunction::new(vertex, FunctionConstantValues::defaults()),
        fragment: RequestFunction::new(fragment, FunctionConstantValues::defaults()),
        template: template.clone(),
        shape: None,
    };
    let compiled = compile_render_request(worker, device, libraries, &request).map(|compiled| {
        (
            compiled
                .result
                .map(|compiled| AcceptancePipeline::Render(Box::new(compiled.pipeline))),
            compiled.measured,
        )
    });
    worker_row(device, compiled, RowStage::Fragment)
}

/// A library's bitcode as a worker input: its bytes, base64 as a retained row stores them.
pub fn retained(bitcode: &[u8]) -> RowSource<'static> {
    RowSource::Retained(Cow::Owned(base64::Engine::encode(
        &base64::engine::general_purpose::STANDARD,
        bitcode,
    )))
}

/// Compile `request` from `libraries` for `device` through the public API (`Library::from_air`,
/// `Library::function`, then `Device::compile_compute_pipeline`, with `compile_function` under
/// the bind's inputs first for a [`PipelineRequest::BoundCompute`], or
/// `Device::compile_render_pipeline` / `compile_render_variant`) under the standard budget with
/// `deadline`. The request's functions come from `libraries` as [`PipelineRequest::libraries`]
/// orders them, each with the functions it links (from its [`PipelineRequest::linked_ranges`]);
/// another number of libraries is [`PipelineCompileError::Libraries`], checked before anything
/// decodes. A [`PipelineRequest::Row`] derives its descriptor and companion from the library's AIR
/// ([`row`]). With the pipeline come the handles of each function's linked functions, from the
/// linked functions the compile built ([`CompiledRequest::handles`]). This is the compile a worker
/// runs ([`crate::worker::compile`]); device tests that own their inputs call it in process.
pub fn compile_pipeline(
    libraries: &[&[u8]],
    request: &PipelineRequest,
    device: &Device,
    deadline: Option<Instant>,
) -> Result<CompiledRequest, PipelineCompileError> {
    let takes = request.libraries();
    if libraries.len() != takes {
        return Err(PipelineCompileError::Libraries {
            takes,
            given: libraries.len(),
        });
    }
    compile_counted(libraries, request, device, deadline)
}

/// [`compile_pipeline`] of `libraries`, which are the ones `request` takes.
fn compile_counted(
    libraries: &[&[u8]],
    request: &PipelineRequest,
    device: &Device,
    deadline: Option<Instant>,
) -> Result<CompiledRequest, PipelineCompileError> {
    let ranges = request.linked_ranges();
    // The libraries the `at`-th function of `request.functions()` links (none for a request
    // whose functions link nothing).
    let linked = |at: usize| ranges.get(at).cloned().unwrap_or(0..0);
    let function = |library: usize, stage: Stage, f: &RequestFunction, linked: Range<usize>| {
        let bitcode = libraries.get(library).ok_or(Error::InvalidDescriptor(
            DescriptorError::MissingStage { stage },
        ))?;
        let function = Library::from_air(bitcode).and_then(|library| library.function(&f.entry))?;
        let (linked, handles) = linked_functions(
            libraries.get(linked).unwrap_or_default(),
            &f.linked,
            &f.constants,
        )?;
        let mut function = FunctionDescriptor::new(function).with_linked(linked);
        function.constants = f.constants.clone();
        Ok::<_, Error>((function, handles))
    };
    // A request whose functions link nothing (a plain compute, an L4 row) answers no handles.
    let unlinked = |pipeline| CompiledRequest {
        pipeline,
        handles: Vec::new(),
    };
    let budget = Budget::with_deadline(deadline);
    let render =
        |descriptor: &RenderPipelineDescriptor, shape: Option<&BindShape>, form| match shape {
            None => device.compile_render_pipeline(descriptor, &budget),
            Some(shape) => crate::candidate::engine::variants::compile_render_for_shape(
                device, descriptor, shape, form, &budget,
            )
            .map(|(pipeline, _)| pipeline),
        };
    Ok(match request {
        PipelineRequest::Compute { entry, constants } => {
            let (kernel, _) = function(
                0,
                Stage::Kernel,
                &RequestFunction::new(entry.as_str(), constants.clone()),
                linked(0),
            )?;
            device
                .compile_compute_pipeline(
                    &ComputePipelineDescriptor::new(StageSource::Function(kernel)),
                    &budget,
                )
                .map(|pipeline| unlinked(AcceptancePipeline::Compute(Box::new(pipeline))))?
        }
        PipelineRequest::BoundCompute {
            function: kernel,
            shape,
            dispatch,
            stage_input,
            tile,
            replay,
        } => {
            let (kernel, handles) = function(0, Stage::Kernel, kernel, linked(0))?;
            let (pipeline, _) = match (tile, stage_input) {
                (None, None) if *replay => {
                    crate::candidate::engine::variants::compile_replay_for_shape(
                        device,
                        &kernel,
                        shape,
                        dispatch.as_ref(),
                        &budget,
                    )?
                }
                _ if *replay => {
                    return Err(PipelineCompileError::Tooling(
                        "a replay variant has no stage-input descriptor and no tile pass".into(),
                    ))
                }
                (Some(formats), None) => {
                    crate::candidate::engine::variants::compile_tile_for_shape(
                        device,
                        &kernel,
                        shape,
                        dispatch.as_ref(),
                        formats,
                        &budget,
                    )?
                }
                (Some(_), Some(_)) => {
                    return Err(PipelineCompileError::Tooling(
                        "a tile pipeline has no stage-input descriptor".into(),
                    ))
                }
                (None, stage_input) => {
                    crate::candidate::engine::variants::compile_compute_for_shape(
                        device,
                        &kernel,
                        shape,
                        dispatch.as_ref(),
                        stage_input.as_ref(),
                        &budget,
                    )?
                }
            };
            CompiledRequest {
                pipeline: AcceptancePipeline::Compute(Box::new(pipeline)),
                handles: vec![handles],
            }
        }
        PipelineRequest::VertexDraw {
            function: vertex,
            template,
            shape,
            form,
        } => {
            let (vertex, handles) = function(0, Stage::Vertex, vertex, linked(0))?;
            let descriptor = template.descriptor(StageSource::Function(vertex), None)?;
            CompiledRequest {
                pipeline: AcceptancePipeline::Render(Box::new(render(
                    &descriptor,
                    shape.as_ref(),
                    *form,
                )?)),
                handles: vec![handles],
            }
        }
        PipelineRequest::Draw {
            vertex,
            fragment,
            template,
            shape,
        } => {
            let (vertex, vertex_handles) = function(0, Stage::Vertex, vertex, linked(0))?;
            let (fragment, fragment_handles) = function(1, Stage::Fragment, fragment, linked(1))?;
            let descriptor = template.descriptor(
                StageSource::Function(vertex),
                Some(StageSource::Function(fragment)),
            )?;
            CompiledRequest {
                pipeline: AcceptancePipeline::Render(Box::new(render(
                    &descriptor,
                    shape.as_ref(),
                    crate::candidate::engine::variants::DrawForm::Bound,
                )?)),
                handles: vec![vertex_handles, fragment_handles],
            }
        }
        PipelineRequest::MeshDraw {
            object,
            mesh,
            fragment,
            template,
            shape,
        } => {
            let mut handles = Vec::new();
            let mut stage = |f: &MeshFunction, stage: Stage| {
                let (function, linked_handles) =
                    function(f.library, stage, &f.function, linked(handles.len()))?;
                handles.push(linked_handles);
                Ok::<_, Error>(StageSource::Function(function))
            };
            let object = object
                .as_ref()
                .map(|object| stage(object, Stage::Object))
                .transpose()?;
            let mesh = stage(mesh, Stage::Mesh)?;
            let fragment = fragment
                .as_ref()
                .map(|fragment| stage(fragment, Stage::Fragment))
                .transpose()?;
            let descriptor = template.descriptor(object, mesh, fragment)?;
            let pipeline = match shape {
                None => device.compile_mesh_pipeline(&descriptor, &budget)?,
                Some(shape) => {
                    crate::candidate::engine::variants::compile_mesh_for_shape(
                        device,
                        &descriptor,
                        shape,
                        &budget,
                    )?
                    .0
                }
            };
            CompiledRequest {
                pipeline: AcceptancePipeline::Render(Box::new(pipeline)),
                handles,
            }
        }
        PipelineRequest::Row {
            entry,
            stage,
            projection,
            constants,
        } => {
            let bitcode = libraries[0];
            let own = |stage| {
                function(
                    0,
                    stage,
                    &RequestFunction::new(entry.as_str(), constants.clone()),
                    0..0,
                )
                .map(|(function, _)| function)
            };
            unlinked(match stage {
                RowStage::Kernel => {
                    let mut descriptor =
                        ComputePipelineDescriptor::new(StageSource::Function(own(Stage::Kernel)?));
                    let compiled = match device.compile_compute_pipeline(&descriptor, &budget) {
                        // A kernel reading stage inputs compiles under the descriptor its AIR
                        // implies, derived only for such a kernel.
                        Err(Error::NeedsInput(RuntimeInput::StageInputDescriptor { .. })) => {
                            descriptor.stage_input =
                                stage_inputs::implied_descriptor(bitcode, entry, constants)
                                    .map_err(|detail| {
                                        PipelineCompileError::Unmodelled(row::Unmodelled {
                                            kind: row::UnmodelledKind::StageInput,
                                            detail,
                                        })
                                    })?;
                            device.compile_compute_pipeline(&descriptor, &budget)
                        }
                        other => other,
                    };
                    compiled.map(|p| AcceptancePipeline::Compute(Box::new(p)))?
                }
                RowStage::Vertex => {
                    // The product decodes the library first, so a module it refuses is its
                    // outcome, not a tool limit of the derivation.
                    let vertex = own(Stage::Vertex)?;
                    let derived = row::vertex_row(bitcode, entry, constants)
                        .map_err(PipelineCompileError::Unmodelled)?;
                    let descriptor = derived
                        .template
                        .descriptor(StageSource::Function(vertex), None)?;
                    let mut variant = VariantKey::new();
                    if *projection == Projection::Pulled {
                        variant.pulled_vertex_buffers = derived.buffers;
                    }
                    device
                        .compile_render_variant(&descriptor, &variant, &budget)
                        .map(|p| AcceptancePipeline::Render(Box::new(p)))?
                }
                RowStage::Object | RowStage::Mesh => {
                    return Err(PipelineCompileError::Unmodelled(row::mesh_pipeline(entry)))
                }
                RowStage::Fragment => {
                    let fragment = own(Stage::Fragment)?;
                    let derived = row::fragment_row(bitcode, entry, constants)
                        .map_err(PipelineCompileError::Unmodelled)?;
                    let companion = derived
                        .companion
                        .opaque_bitcode()
                        .map_err(PipelineCompileError::Tooling)?;
                    let vertex = Library::from_air(&companion)
                        .and_then(|library| library.function(derived.vertex_entry))
                        .map(FunctionDescriptor::new)?;
                    let descriptor = derived.template.descriptor(
                        StageSource::Function(vertex),
                        Some(StageSource::Function(fragment)),
                    )?;
                    device
                        .compile_render_pipeline(&descriptor, &budget)
                        .map(|p| AcceptancePipeline::Render(Box::new(p)))?
                }
            })
        }
    })
}

/// The product's `LinkedFunctions` of a function with `constants` that links `linked`, whose
/// bitcode is `libraries`, one each, in the same order: each library decoded once, each of its
/// functions in `functions` (the list the Metal oracle puts every linked function into), made with
/// the function's constants as `newFunctionWithName:constantValues:` makes it: the compile takes
/// the values of the constants its library declares and ignores the others, as Metal does. With
/// them come their table handles ([`LinkedHandles`]): each function's
/// `LinkedFunctions::function_handle` over the whole list.
fn linked_functions(
    libraries: &[&[u8]],
    linked: &[LinkedLibrary],
    constants: &FunctionConstantValues,
) -> Result<(LinkedFunctions, LinkedHandles), Error> {
    let mut functions = LinkedFunctions::new();
    for (library, bitcode) in linked.iter().zip(libraries) {
        let decoded = Library::from_air(bitcode)?;
        for name in &library.functions {
            let function = decoded.function(name)?;
            functions = functions
                .with_function(LinkedFunction::new(function).with_constants(constants.clone()));
        }
    }
    // `functions` lists each library's functions in turn, in `linked`'s order.
    let mut listed = functions.functions.iter();
    let handles = linked
        .iter()
        .map(|library| {
            listed
                .by_ref()
                .take(library.functions.len())
                .map(|function| functions.function_handle(function))
                .collect()
        })
        .collect();
    Ok((functions, handles))
}

/// What the driver did with one pipeline.
#[derive(Debug)]
pub enum Acceptance {
    /// The driver created the pipeline; `create` is the `vkCreate*Pipelines` call alone, by a
    /// `budgets::CreationClock`.
    Created { create: Duration },
    /// The device did not enable (or does not provide) something the pipeline requires, or a
    /// requirement could not be checked against it; creating it could be invalid usage, so
    /// nothing was created.
    NotEnabled(Vec<String>),
    /// The driver refused the pipeline (or one of its objects).
    Refused(String),
    /// `VK_LAYER_KHRONOS_validation` reported invalid usage while the pipeline was created:
    /// whatever the driver did, the row is not evidence.
    InvalidUsage(Vec<String>),
}

/// Why an acceptance attempt gave no verdict about a pipeline. The kinds are kept apart because
/// a sweep treats them differently: an unavailable device ends it, a modelled limit of the
/// creation path is one row the path does not build, and any other tooling error fails it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AcceptanceError {
    /// The Vulkan context could not be opened or used (no implementation of the requested type,
    /// the hang tripwire set). No row can be attempted on it, so a sweep stops with an error
    /// rather than report every row as untried and pass.
    DeviceUnavailable(String),
    /// A state the creation path does not build (a per-bind stride the pipeline keeps static,
    /// which a consumer creates per stride; a depth and a stencil format no one attachment
    /// holds): a modelled limit of the tool, never a verdict
    /// (`candidate::platform::graphics_pipeline`).
    Unmodelled(String),
    /// Anything else the tool could not do for this row (inspecting a module, an entry-point
    /// name): a tool defect, which fails the sweep.
    Tooling(String),
}

impl std::fmt::Display for AcceptanceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AcceptanceError::DeviceUnavailable(why) => write!(f, "device unavailable: {why}"),
            AcceptanceError::Unmodelled(why) => f.write_str(why),
            AcceptanceError::Tooling(why) => f.write_str(why),
        }
    }
}

/// What of `requirements` the opened device did not enable, one name each (the `Debug` of the
/// `Capability` the product names): the API version, each feature (a stage create flag's
/// feature) with its exposing extension, each device extension, and each format's defining
/// extension, every one checked alone with the product's `DeviceDescription::required_enables`
/// against `opened`, the device's queried description restricted to what the context enabled
/// (`DeviceDescription::with_enabled`). Creating a pipeline that needs any of them would be invalid
/// usage, so the row is reported as not enabled rather than accepted or refused. When the opened
/// description could not be built, a pipeline that needs any enable is unchecked and reported as
/// such, never created.
///
/// Properties, limits, format support, and float-controls independence are
/// [`unmet_device_facts`]'s.
pub fn unenabled(
    opened: &Result<DeviceDescription, String>,
    requirements: &Requirements,
) -> Vec<String> {
    let mut singles = Vec::new();
    if let Some(version) = requirements.api_version {
        let mut single = Requirements::default();
        single.api_version = Some(version);
        singles.push(single);
    }
    for &feature in &requirements.features {
        let mut single = Requirements::default();
        single.features.insert(feature);
        singles.push(single);
    }
    for &extension in &requirements.extensions {
        let mut single = Requirements::default();
        single.extensions.insert(extension);
        singles.push(single);
    }
    for &format in &requirements.formats {
        let mut single = Requirements::default();
        single.formats.insert(format);
        singles.push(single);
    }
    for &flag in &requirements.stage_flags {
        let mut single = Requirements::default();
        single.stage_flags.insert(flag);
        singles.push(single);
    }
    if singles.is_empty() {
        return Vec::new();
    }
    let description = match opened {
        Ok(description) => description,
        Err(error) => return vec![format!("unchecked enables: {error}")],
    };
    let mut missing: Vec<String> = Vec::new();
    for single in &singles {
        if let Err(capability) = description.required_enables(single) {
            let name = format!("{capability:?}");
            if !missing.contains(&name) {
                missing.push(name);
            }
        }
    }
    missing
}

/// The profile of the opened device: `description` (what the physical device supports, as
/// `DeviceDescription::query` read it) restricted to what the context enabled
/// (`DeviceDescription::with_enabled`, which refuses an enable set the description does not
/// list or that is not closed), derived under the default policy the committed profiles use.
/// Its properties, limits, formats, and float controls are the device's own queried facts, so a
/// committed profile that overstates the device cannot vouch for a pipeline.
pub fn opened_profile(
    description: DeviceDescription,
    enabled: &EnableSet,
) -> Result<DeviceProfile, String> {
    opened_description(description, enabled).and_then(derived_profile)
}

/// `description` restricted to what the context enabled (`DeviceDescription::with_enabled`). The
/// engine candidate executor derives its `Device` from this.
pub fn opened_description(
    description: DeviceDescription,
    enabled: &EnableSet,
) -> Result<DeviceDescription, String> {
    description
        .with_enabled(enabled)
        .map_err(|error| format!("restrict the device's description to what it enabled: {error}"))
}

/// `description` as a context created at API version `instance` reads a device that reports it
/// (`DeviceDescription::query`'s rule): the version is the lower of the two, a feature that
/// version does not expose with the listed extensions is not read, and only the formats the
/// product asks about are ([`FormatSupport::queries_for`], which a profile document's other
/// formats, such as the scaled vertex formats, are not among).
///
/// [`FormatSupport::queries_for`]: metal2vulkan::device::FormatSupport::queries_for
pub fn read_at(
    description: &DeviceDescription,
    instance: metal2vulkan::device::ApiVersion,
) -> DeviceDescription {
    let mut read = description.clone();
    let api = read.api_version.min(instance);
    read.api_version = api;
    let extensions = read.extensions.clone();
    read.features
        .retain(|&feature| feature.exposure().exposed(api, &extensions));
    let queried: std::collections::BTreeSet<_> =
        metal2vulkan::device::FormatSupport::queries_for(api, &extensions)
            .into_iter()
            .collect();
    read.formats
        .formats
        .retain(|format, _| queried.contains(format));
    read
}

/// The profile of an opened description ([`opened_description`]) under the default policy.
pub fn derived_profile(description: DeviceDescription) -> Result<DeviceProfile, String> {
    Device::new(description, DevicePolicy::default())
        .map(|device| device.profile().clone())
        .map_err(|error| format!("derive the opened device's profile: {error}"))
}

/// Whether `requirements` needs anything beyond the API version, features, and extensions.
fn needs_device_facts(requirements: &Requirements) -> bool {
    !requirements.properties.is_empty()
        || !requirements.limits.is_empty()
        || !requirements.formats.is_empty()
        || !requirements.stage_flags.is_empty()
        || requirements.denorm_independence != Independence::None
        || requirements.rounding_independence != Independence::None
}

/// The properties, limits, formats, stage create flags, and float-controls independence of
/// `requirements` the opened device does not provide, one name each, checked with the product's
/// own `DeviceProfile::check` against `opened` ([`opened_profile`]). When the opened device's
/// profile could not be derived, a pipeline that needs any of them is unchecked and reported as
/// such (never created): the committed profile it was compiled for is not evidence about the
/// device.
pub fn unmet_device_facts(
    opened: &Result<DeviceProfile, String>,
    requirements: &Requirements,
) -> Vec<String> {
    if !needs_device_facts(requirements) {
        return Vec::new();
    }
    let profile = match opened {
        Ok(profile) => profile,
        Err(error) => return vec![format!("unchecked device facts: {error}")],
    };
    // One requirement at a time, so every unmet one is named (the check stops at the first).
    let mut singles = Vec::new();
    for &property in &requirements.properties {
        let mut single = Requirements::default();
        single.properties.insert(property);
        singles.push(single);
    }
    for (&limit, &value) in &requirements.limits {
        let mut single = Requirements::default();
        single.limits.insert(limit, value);
        singles.push(single);
    }
    for &format in &requirements.formats {
        let mut single = Requirements::default();
        single.formats.insert(format);
        singles.push(single);
    }
    for &flag in &requirements.stage_flags {
        let mut single = Requirements::default();
        single.stage_flags.insert(flag);
        singles.push(single);
    }
    let mut single = Requirements::default();
    single.denorm_independence = requirements.denorm_independence;
    singles.push(single);
    let mut single = Requirements::default();
    single.rounding_independence = requirements.rounding_independence;
    singles.push(single);
    let mut unmet: Vec<String> = Vec::new();
    for single in &singles {
        if let Err(capability) = profile.check(single) {
            let name = format!("{capability:?}");
            if !unmet.contains(&name) {
                unmet.push(name);
            }
        }
    }
    unmet
}

#[cfg(test)]
mod tests {
    use super::*;
    use metal2vulkan::library::FunctionConstantValue;
    use metal2vulkan::vk;
    use metal2vulkan_devtools::{engine_dir, llvm_assemble, AssembleError};

    fn fixture(name: &str) -> Option<Vec<u8>> {
        match llvm_assemble(&engine_dir().join(name)) {
            Ok(bytes) => Some(bytes),
            Err(AssembleError::Missing(why)) => {
                eprintln!("SKIPPED: {why}");
                None
            }
            Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {name}: {why}"),
        }
    }

    /// A kernel row reading stage inputs compiles under the stage-input descriptor its AIR implies
    /// (each attribute its own buffer at the executors' slot), and the pipeline binds those
    /// buffers.
    #[test]
    fn a_stage_input_kernel_row_compiles_under_its_implied_descriptor() {
        let Some(bitcode) = fixture("kernel_stage_in.ll") else {
            return;
        };
        let device = crate::engine::load_device("moltenvk-m3").unwrap();
        let undefined = FunctionConstantValues::defaults();
        let implied = stage_inputs::implied_descriptor(&bitcode, "stage_in_fetch", &undefined)
            .unwrap()
            .expect("a descriptor");
        assert_eq!(implied.attributes.len(), 4);
        let RowCompile::Compiled(pipeline) = compile_row(
            &bitcode,
            "stage_in_fetch",
            RowStage::Kernel,
            Projection::Derived,
            &device,
            None,
        ) else {
            panic!("stage_in_fetch did not compile");
        };
        let RowPipeline::Compute {
            pipeline: compute, ..
        } = pipeline.as_ref()
        else {
            panic!("a kernel compiles to a compute pipeline");
        };
        for a in implied.attributes.values() {
            assert!(
                compute
                    .interface
                    .slots
                    .keys()
                    .any(|slot| slot.index == a.buffer_index),
                "{a:?}"
            );
        }
        assert_eq!(
            stage_inputs::implied_descriptor(&bitcode, "stage_in_region_short", &undefined)
                .unwrap(),
            None
        );
    }

    #[test]
    fn a_kernel_row_compiles_to_a_compute_pipeline_carrying_its_binder_layout() {
        let Some(bitcode) = fixture("store_const.ll") else {
            return;
        };
        let device = crate::engine::load_device("vk12-floor").unwrap();
        let RowCompile::Compiled(pipeline) = compile_row(
            &bitcode,
            "store_const",
            RowStage::Kernel,
            Projection::Derived,
            &device,
            None,
        ) else {
            panic!("store_const did not compile");
        };
        let RowPipeline::Compute {
            pipeline: compute, ..
        } = pipeline.as_ref()
        else {
            panic!("a kernel compiles to a compute pipeline");
        };
        assert_eq!(compute.stage.entry_point, "main");
        assert!(pipeline.words() > 5);
        // The layout the driver sees is dense set 0.
        let set0 = &compute.layout.sets[0];
        assert_eq!(set0.set, 0);
        for (index, binding) in set0.bindings.iter().enumerate() {
            assert_eq!(binding.binding as usize, index);
        }
        // Every requirement is enabled, and every device fact met, on a device that is exactly
        // the profile's description and enabled everything it recommends.
        let description = description("vk12-floor");
        let enabled = description.recommended_enables(&DevicePolicy::default());
        assert_eq!(
            unenabled(
                &opened_description(description.clone(), &enabled),
                pipeline.requirements()
            ),
            Vec::<String>::new()
        );
        let opened = opened_profile(description.clone(), &enabled);
        assert_eq!(
            unmet_device_facts(&opened, pipeline.requirements()),
            Vec::<String>::new()
        );

        let row = |bitcode: &[u8], entry, stage| {
            compile_row(bitcode, entry, stage, Projection::Derived, &device, None)
        };
        match row(&bitcode, "no_such_entry", RowStage::Kernel) {
            RowCompile::NotCompiled(outcome, detail) => {
                assert_eq!(outcome.label(), "invalid_descriptor", "{detail}");
            }
            other => panic!("{other:?}"),
        }
        // A kernel is no fragment function: the product refuses the draw's descriptor.
        match row(&bitcode, "store_const", RowStage::Fragment) {
            RowCompile::NotCompiled(outcome, detail) => {
                assert_eq!(outcome.label(), "invalid_descriptor", "{detail}");
            }
            other => panic!("{other:?}"),
        }
        match row(b"not bitcode", "main", RowStage::Kernel) {
            RowCompile::NotCompiled(outcome, _) => assert_eq!(outcome.label(), "invalid_input"),
            other => panic!("{other:?}"),
        }
    }

    /// A vertex row that reads attributes compiles against the descriptor the tool synthesizes
    /// for them (fixed-function input where the device fetches the format, or every attribute
    /// pulled); a fragment row compiles into a rasterizing pipeline with its generated
    /// passthrough vertex companion and the attachments its outputs imply.
    #[test]
    fn vertex_and_fragment_rows_compile_with_their_derived_inputs() {
        let Some(bitcode) = fixture("stage_io.ll") else {
            return;
        };
        let device = crate::engine::load_device("lavapipe").unwrap();
        let row = |entry, stage, projection| match compile_row(
            &bitcode, entry, stage, projection, &device, None,
        ) {
            RowCompile::Compiled(pipeline) => match *pipeline {
                RowPipeline::Render { pipeline, .. } => pipeline,
                other => panic!("{entry} compiled to {other:?}"),
            },
            other => panic!("{entry} {projection:?}: {other:?}"),
        };
        let vertex = row("stage_io_vertex", RowStage::Vertex, Projection::Derived);
        assert!(vertex.graphics.rasterizer_discard);
        assert_eq!(vertex.vertex_input.attributes.len(), 4);
        assert_eq!(vertex.vertex_input.bindings.len(), 4);
        let pulled = row("stage_io_vertex", RowStage::Vertex, Projection::Pulled);
        assert!(pulled.vertex_input.attributes.is_empty(), "{pulled:?}");
        assert!(pulled.vertex_input.bindings.is_empty());
        assert_ne!(pulled.stages[0].spirv, vertex.stages[0].spirv);

        let fragment = row("stage_io_fragment", RowStage::Fragment, Projection::Derived);
        assert!(!fragment.graphics.rasterizer_discard);
        assert_eq!(fragment.stages.len(), 2);
        let colors: Vec<usize> = fragment
            .graphics
            .color_attachments
            .iter()
            .enumerate()
            .filter_map(|(index, state)| state.map(|_| index))
            .collect();
        assert_eq!(colors, [0, 2]);
    }

    /// A row compiled with function-constant values (`corpus-pipeline --specializations`) derives
    /// its pipeline inputs under them and compiles under them: a depth output a constant gates has
    /// its aspect only where the values leave the output (the stencil output beside it keeps its
    /// own), and a kernel that uses a constant's value, which needs the value when the constant is
    /// undefined, compiles once the values supply it.
    #[test]
    fn a_row_with_constants_derives_its_inputs_and_compiles_under_them() {
        let ll = crate::executor_contract::depth_stencil::tests::depth_stencil_module(
            Some("any"),
            true,
            true,
        );
        let gated = match crate::check::loop_free::assemble(&ll) {
            Ok(bitcode) => bitcode,
            Err(error) if error.contains("llvm-as is unavailable") => {
                eprintln!("SKIPPED: {error}");
                return;
            }
            Err(error) => panic!("{error}\n{ll}"),
        };
        let Some(vector) = fixture("kernel_vector_function_constant.ll") else {
            return;
        };
        let flag = |value: u8| {
            FunctionConstantValues::defaults().with_index(
                0,
                FunctionConstantValue::new(metal2vulkan::metal::DataType::Bool, vec![value]),
            )
        };
        let aspects = |constants: &FunctionConstantValues| {
            let template = row::fragment_row(&gated, "frag", constants)
                .unwrap()
                .template;
            (template.depth_format, template.stencil_format)
        };
        let stencil = (None, Some(PixelFormat::Stencil8));
        assert_eq!(aspects(&FunctionConstantValues::defaults()), stencil);
        assert_eq!(aspects(&flag(0)), stencil);
        let both = Some(PixelFormat::Depth32FloatStencil8);
        assert_eq!(aspects(&flag(1)), (both, both));

        let device = crate::engine::load_device("lavapipe").unwrap();
        let kernel = |constants| {
            let request = PipelineRequest::Row {
                entry: "kernel_vector_function_constant".into(),
                stage: RowStage::Kernel,
                projection: Projection::Derived,
                constants,
            };
            compile_pipeline(&[&vector], &request, &device, None).map(|c| c.pipeline)
        };
        // Undefined, the constant reads its type's zero: Metal makes this pipeline too.
        match kernel(FunctionConstantValues::defaults()) {
            Ok(AcceptancePipeline::Compute(_)) => {}
            other => panic!("{other:?}"),
        }
        let lanes = [1u32, 2, 3, 4].iter().flat_map(|lane| lane.to_le_bytes());
        let value = FunctionConstantValue::new(
            metal2vulkan::metal::DataType::UInt4,
            lanes.collect::<Vec<u8>>(),
        );
        match kernel(FunctionConstantValues::defaults().with_index(0, value)) {
            Ok(AcceptancePipeline::Compute(_)) => {}
            other => panic!("{other:?}"),
        }
    }

    /// A row's words are its own function's stage, the module the translation sweep
    /// (`engine::translate_new`) compiles and counts for it; a fragment row's generated passthrough
    /// vertex companion and a post-tessellation vertex row's synthesized stages are counted apart,
    /// and a kernel or a vertex function alone has no other stage. The sweep translates a
    /// post-tessellation vertex function alone where it reads no control-point attributes (its
    /// words are the row's evaluation stage's), and refuses one that does (`needs_input`: the
    /// descriptor of its patch inputs, which the row derives).
    #[test]
    fn a_rows_words_are_the_module_the_translation_sweep_counts() {
        let (Some(stage_io), Some(store_const), Some(tessellation)) = (
            fixture("stage_io.ll"),
            fixture("store_const.ll"),
            fixture("tessellation.ll"),
        ) else {
            return;
        };
        let device = crate::engine::load_device("lavapipe").unwrap();
        let profile = crate::engine::load_profile("lavapipe").unwrap();
        /// What the translation sweep has for the row's function compiled alone.
        #[derive(Clone, Copy)]
        enum Swept {
            Words,
            NeedsInput,
        }
        for (bitcode, entry, stage, companion, expect) in [
            (
                &store_const,
                "store_const",
                RowStage::Kernel,
                false,
                Swept::Words,
            ),
            (
                &stage_io,
                "stage_io_vertex",
                RowStage::Vertex,
                false,
                Swept::Words,
            ),
            (
                &stage_io,
                "stage_io_fragment",
                RowStage::Fragment,
                true,
                Swept::Words,
            ),
            // A post-tessellation vertex function's own stage is its evaluation stage; the
            // synthesized vertex and control stages are counted apart.
            (
                &tessellation,
                "tess_quad_buffer",
                RowStage::Vertex,
                true,
                Swept::Words,
            ),
            (
                &tessellation,
                "tess_short_ids",
                RowStage::Vertex,
                true,
                Swept::Words,
            ),
            (
                &tessellation,
                "tess_triangle_points",
                RowStage::Vertex,
                true,
                Swept::NeedsInput,
            ),
        ] {
            let compiled = compile_row(bitcode, entry, stage, Projection::Derived, &device, None);
            let RowCompile::Compiled(row) = compiled else {
                panic!("{entry}: {compiled:?}");
            };
            let swept = crate::engine::translate_new(bitcode, entry, &profile, None);
            if let Swept::Words = expect {
                assert_eq!(Some(row.words()), swept.words, "{entry}");
            } else {
                // The function reads control-point attributes: the sweep compiles it without
                // the descriptor of its patch inputs the row derives, and has no words.
                assert_eq!(swept.outcome.label(), "needs_input", "{entry}");
                assert_eq!(swept.words, None, "{entry}");
            }
            if entry.starts_with("tess_") {
                assert_eq!(
                    row.own_stage(),
                    vk::ShaderStage::TESSELLATION_EVALUATION,
                    "{entry}"
                );
            }
            let own = row.own_stage();
            let others: u64 = row
                .stages()
                .iter()
                .filter(|module| module.stage != own)
                .map(|module| module.spirv.len() as u64)
                .sum();
            assert_eq!(row.companion_words(), others, "{entry}");
            assert_eq!(row.companion_words() > 0, companion, "{entry}");
        }
    }

    /// A fragment row whose outputs have fewer than four components compiles: its attachments have
    /// no more channels than its outputs write, which the product, as Metal, requires (a `uint2`
    /// target into `RGBA32Uint` was `InvalidDescriptor(ColorAttachment { index: 2, .. })`).
    #[test]
    fn a_fragment_row_with_narrow_outputs_compiles_into_its_derived_attachments() {
        let Some(bitcode) = fixture("signed_color.ll") else {
            return;
        };
        let device = crate::engine::load_device("moltenvk-m3").unwrap();
        let RowCompile::Compiled(pipeline) = compile_row(
            &bitcode,
            "signed_targets_f",
            RowStage::Fragment,
            Projection::Derived,
            &device,
            None,
        ) else {
            panic!("signed_targets_f did not compile");
        };
        let RowPipeline::Render {
            pipeline: render, ..
        } = pipeline.as_ref()
        else {
            panic!("a fragment row compiles to a render pipeline");
        };
        let formats: Vec<(usize, vk::Format)> = render
            .graphics
            .color_attachments
            .iter()
            .enumerate()
            .filter_map(|(index, state)| state.map(|state| (index, state.format)))
            .collect();
        assert_eq!(
            formats,
            [
                (0, vk::Format::R32G32B32A32_SFLOAT),
                (1, vk::Format::R32G32B32A32_UINT),
                (2, vk::Format::R32G32_UINT),
            ]
        );
    }

    /// A post-tessellation vertex row compiles to a tessellation pipeline: the synthesized vertex
    /// and control stages before the function's evaluation stage, a patch list of one control point
    /// per patch, and the row's control-point and patch attributes pulled by the evaluation stage.
    /// On a device without tessellation shaders it compiles to the emulation: the function as the
    /// one vertex stage, drawing a triangle list, its attributes pulled; on one whose vertex stage
    /// cannot bind the emulation's buffers beside the function's (the floor's four storage
    /// buffers), it is refused typed.
    #[test]
    fn post_tessellation_vertex_rows_compile_to_tessellation_pipelines() {
        let Some(bitcode) = fixture("tessellation.ll") else {
            return;
        };
        let compile = |profile: &str, entry| {
            let device = crate::engine::load_device(profile).unwrap();
            compile_row(
                &bitcode,
                entry,
                RowStage::Vertex,
                Projection::Derived,
                &device,
                None,
            )
        };
        for entry in ["tess_quad_buffer", "tess_triangle_points"] {
            let RowCompile::Compiled(pipeline) = compile("lavapipe", entry) else {
                panic!("{entry} did not compile");
            };
            let RowPipeline::Render {
                pipeline: render, ..
            } = *pipeline
            else {
                panic!("{entry} is a render pipeline");
            };
            let stages: Vec<_> = render.stages.iter().map(|stage| stage.stage).collect();
            assert_eq!(
                stages,
                [
                    vk::ShaderStage::VERTEX,
                    vk::ShaderStage::TESSELLATION_CONTROL,
                    vk::ShaderStage::TESSELLATION_EVALUATION
                ],
                "{entry}"
            );
            let tessellation = render.graphics.tessellation.expect("tessellation state");
            assert_eq!(tessellation.patch_control_points, 1);
            assert!(render.graphics.rasterizer_discard);
            // The fixed vertex stage reads no attribute: every one is pulled.
            assert!(render.vertex_input.attributes.is_empty(), "{entry}");
        }
        for entry in ["tess_quad_buffer", "tess_triangle_points"] {
            let RowCompile::Compiled(pipeline) = compile("roadmap-2022", entry) else {
                panic!("{entry} did not compile on roadmap-2022");
            };
            let RowPipeline::Render {
                pipeline: render, ..
            } = *pipeline
            else {
                panic!("{entry} is a render pipeline");
            };
            let stages: Vec<_> = render.stages.iter().map(|stage| stage.stage).collect();
            assert_eq!(stages, [vk::ShaderStage::VERTEX], "{entry}");
            assert_eq!(render.graphics.tessellation, None, "{entry}");
            assert!(render.graphics.emulated_tessellation.is_some(), "{entry}");
            assert_eq!(
                render.graphics.topology,
                vk::PrimitiveTopology::TRIANGLE_LIST,
                "{entry}"
            );
            assert!(render.vertex_input.attributes.is_empty(), "{entry}");
        }
        match compile("vk12-floor", "tess_triangle_points") {
            RowCompile::NotCompiled(outcome, detail) => {
                assert_eq!(outcome.label(), "device_capability_missing", "{detail}");
                assert!(
                    detail.contains("MaxPerStageDescriptorStorageBuffers"),
                    "{detail}"
                );
            }
            other => panic!("{other:?}"),
        }
    }

    /// A request function links the functions of its linked libraries: the caller of the public
    /// linked fixtures compiles only with `linked_scale` and `linked_blend` of the functions
    /// library and `linked_offset` of its own, the libraries following the stage library in request
    /// order. With the pipeline come the product's table handles of those functions, by library and
    /// function in request order: `functions` numbered from 2. A compile given fewer or more
    /// libraries than the request takes is the tool's exchange error before anything decodes, never
    /// a product outcome.
    #[test]
    fn a_request_function_links_the_functions_of_its_libraries() {
        let (Some(caller), Some(functions)) = (
            fixture("linked_visible_caller.ll"),
            fixture("linked_visible_functions.ll"),
        ) else {
            return;
        };
        let device = crate::engine::load_device("moltenvk-m3").unwrap();
        let request = |linked: Vec<LinkedLibrary>| PipelineRequest::BoundCompute {
            function: RequestFunction {
                entry: "linked_visible_caller".into(),
                constants: FunctionConstantValues::defaults(),
                linked,
            },
            shape: BindShape::default(),
            dispatch: None,
            stage_input: None,
            tile: None,
            replay: false,
        };
        let library = |names: &[&str]| LinkedLibrary {
            functions: names.iter().map(|name| (*name).to_owned()).collect(),
        };
        let linked = request(vec![
            library(&["linked_scale", "linked_blend"]),
            library(&["linked_offset"]),
        ]);
        assert_eq!(linked.stage_libraries(), 1);
        assert_eq!(linked.libraries(), 3);
        let libraries = [caller.as_slice(), functions.as_slice(), caller.as_slice()];
        match compile_pipeline(&libraries, &linked, &device, None) {
            Ok(CompiledRequest {
                pipeline: AcceptancePipeline::Compute(pipeline),
                handles,
            }) => {
                assert!(pipeline.stage.spirv.len() > 5);
                linked.check_handles(&handles).unwrap();
                let words = handles
                    .iter()
                    .map(|function| {
                        function
                            .iter()
                            .map(|library| {
                                library
                                    .iter()
                                    .map(|h| h.map(FunctionHandle::word))
                                    .collect()
                            })
                            .collect()
                    })
                    .collect::<Vec<Vec<Vec<_>>>>();
                assert_eq!(words, [[vec![Some(2), Some(3)], vec![Some(4)]]]);
                assert!(linked.check_handles(&[]).is_err());
                assert!(linked
                    .check_handles(&[vec![handles[0][0].clone()]])
                    .is_err());
            }
            other => panic!("the linked caller did not compile: {other:?}"),
        }
        assert_eq!(
            compile_pipeline(&[&caller], &request(Vec::new()), &device, None).unwrap_err(),
            PipelineCompileError::Product(Error::NeedsInput(
                metal2vulkan::error::RuntimeInput::LinkedFunction {
                    symbol: metal2vulkan::error::DiagName::new("linked_scale"),
                }
            ))
        );
        let with = [caller.as_slice(); 4];
        for given in [2, 4] {
            assert_eq!(
                compile_pipeline(&with[..given], &linked, &device, None).unwrap_err(),
                PipelineCompileError::Libraries { takes: 3, given }
            );
        }
        let one = request(Vec::new());
        assert_eq!(
            compile_pipeline(&libraries, &one, &device, None).unwrap_err(),
            PipelineCompileError::Libraries { takes: 1, given: 3 }
        );
    }

    /// The linked libraries of each function follow the stage libraries in stage order: a draw's
    /// vertex function links the libraries right after the two stage libraries, its fragment
    /// function those right after the vertex function's.
    #[test]
    fn each_function_links_the_libraries_after_the_previous_ones() {
        let function = |entry: &str, libraries: usize| RequestFunction {
            entry: entry.into(),
            constants: FunctionConstantValues::defaults(),
            linked: vec![LinkedLibrary::default(); libraries],
        };
        let template = crate::candidate::engine::graphics::RenderTemplate::new(
            metal2vulkan::metal::PrimitiveTopologyClass::Triangle,
        );
        let draw = |vertex: usize, fragment: usize| PipelineRequest::Draw {
            vertex: function("v", vertex),
            fragment: function("f", fragment),
            template: template.clone(),
            shape: None,
        };
        assert_eq!(draw(2, 1).linked_ranges(), [2..4, 4..5]);
        assert_eq!(draw(2, 1).libraries(), 5);
        assert_eq!(draw(0, 3).linked_ranges(), [2..2, 2..5]);
        assert_eq!(draw(1, 0).linked_ranges(), [2..3, 3..3]);
        let kernel = PipelineRequest::BoundCompute {
            function: function("k", 2),
            shape: BindShape::default(),
            dispatch: None,
            stage_input: None,
            tile: None,
            replay: false,
        };
        assert_eq!(kernel.linked_ranges(), vec![Range { start: 1, end: 3 }]);
        assert_eq!(kernel.libraries(), 3);
        let vertex = PipelineRequest::VertexDraw {
            function: function("v", 1),
            template,
            shape: None,
            form: crate::candidate::engine::variants::DrawForm::Bound,
        };
        assert_eq!(vertex.linked_ranges(), vec![Range { start: 1, end: 2 }]);
        let row = PipelineRequest::Compute {
            entry: "k".into(),
            constants: FunctionConstantValues::defaults(),
        };
        assert!(row.linked_ranges().is_empty());
        assert_eq!(row.libraries(), 1);
    }

    /// Each linked function is made with the request function's constants, as the Metal oracle
    /// makes it with `newFunctionWithName:constantValues:`: every value, those of constants its
    /// library does not declare among them, which the compile ignores as Metal does. The table
    /// handles answer the libraries' functions in order.
    #[test]
    fn a_linked_function_is_made_with_the_request_functions_constants() {
        let (Some(declares), Some(declares_none)) = (
            fixture("kernel_vector_function_constant.ll"),
            fixture("linked_visible_functions.ll"),
        ) else {
            return;
        };
        let uint4 = |value: u32| {
            FunctionConstantValue::new(
                metal2vulkan::metal::DataType::UInt4,
                [value; 4]
                    .iter()
                    .flat_map(|lane| lane.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
        };
        let constants = FunctionConstantValues::defaults()
            .with_index(0, uint4(1))
            .with_index(3, uint4(2))
            .with_name("vector_value", uint4(3))
            .with_name("undeclared", uint4(4));
        let libraries = [declares.as_slice(), declares_none.as_slice()];
        let linked = [
            LinkedLibrary {
                functions: vec!["kernel_vector_function_constant".into()],
            },
            LinkedLibrary {
                functions: vec!["linked_scale".into(), "linked_blend".into()],
            },
        ];
        let (functions, handles) = linked_functions(&libraries, &linked, &constants).unwrap();
        // Each function's table handle, by library and function.
        let words = handles
            .iter()
            .map(|library| {
                library
                    .iter()
                    .map(|h| h.map(FunctionHandle::word))
                    .collect()
            })
            .collect::<Vec<Vec<_>>>();
        assert_eq!(words, [vec![Some(2)], vec![Some(3), Some(4)]]);
        let taken = functions
            .functions
            .iter()
            .map(|f| (f.function.info().name.to_owned(), f.constants.clone()))
            .collect::<Vec<_>>();
        assert_eq!(
            taken,
            [
                (
                    "kernel_vector_function_constant".to_owned(),
                    constants.clone()
                ),
                ("linked_scale".to_owned(), constants.clone()),
                ("linked_blend".to_owned(), constants.clone()),
            ]
        );
        assert!(functions.private_functions.is_empty() && functions.groups.is_empty());
    }

    /// A worker that cannot start is a stop at the boundary, never a compile outcome, for every
    /// stage.
    #[test]
    fn a_worker_boundary_failure_is_a_stopped_row() {
        let device = WorkerDevice::new(description("vk12-floor"), DevicePolicy::default()).unwrap();
        let worker = CompileWorker::program("/nonexistent/compile-worker");
        for stage in [RowStage::Kernel, RowStage::Vertex, RowStage::Fragment] {
            let (row, measured) = compile_row_in_worker(
                &worker,
                &device,
                &retained(b"bitcode"),
                "main",
                stage,
                Projection::Derived,
                &FunctionConstantValues::defaults(),
            );
            match row {
                RowCompile::Stopped(outcome, detail) => {
                    assert_eq!(outcome.key(), "tooling", "{detail}");
                    assert!(detail.contains("spawn"), "{detail}");
                }
                other => panic!("{other:?}"),
            }
            assert_eq!(measured, None);
        }
    }

    /// Every requirement the opened description lacks is named once, whatever needs it (a stage
    /// flag's feature is the feature's name); nothing is checked without a description.
    #[test]
    fn unenabled_names_every_requirement_the_opened_device_did_not_enable() {
        use metal2vulkan::device::{ApiVersion, StageCreateFlag, VkExtension, VkFeature};
        let description = description("vk12-floor");
        let mut enabled = EnableSet::new();
        enabled.features.insert(VkFeature::RobustBufferAccess);
        let opened = opened_description(description, &enabled);
        assert!(opened.is_ok(), "{opened:?}");

        let mut requirements = Requirements::default();
        requirements.api_version = Some(ApiVersion { major: 1, minor: 3 });
        requirements.features.insert(VkFeature::RobustBufferAccess);
        requirements.features.insert(VkFeature::ShaderInt64);
        requirements.extensions.insert(VkExtension::ExtRobustness2);
        requirements
            .stage_flags
            .insert(StageCreateFlag::RequireFullSubgroups);
        assert_eq!(
            unenabled(&opened, &requirements),
            [
                "ApiVersion(ApiVersion { major: 1, minor: 3 })",
                "Feature(ShaderInt64)",
                "Extension(ExtRobustness2)",
                "Feature(ComputeFullSubgroups)",
            ]
        );
        // The flag's feature named again is named once.
        requirements
            .features
            .insert(VkFeature::ComputeFullSubgroups);
        assert_eq!(unenabled(&opened, &requirements).len(), 4);
        assert!(unenabled(&opened, &Requirements::default()).is_empty());

        let failed: Result<DeviceDescription, String> = Err("no description".into());
        assert_eq!(
            unenabled(&failed, &requirements),
            ["unchecked enables: no description"]
        );
        assert!(unenabled(&failed, &Requirements::default()).is_empty());
    }

    /// A render row compiled for a profile whose pipelines make the extended dynamic states
    /// dynamic requires `extendedDynamicState`. A device opened at API 1.2 without the feature
    /// enabled names it as not enabled, so the row is never created; the same device with the
    /// feature and its extension enabled names nothing of it.
    #[test]
    fn a_render_row_names_extended_dynamic_state_the_device_did_not_enable() {
        use metal2vulkan::device::{ApiVersion, VkFeature};
        let Some(bitcode) = fixture("stage_io.ll") else {
            return;
        };
        let device = crate::engine::load_device("lavapipe").unwrap();
        let pipeline = match compile_row(
            &bitcode,
            "stage_io_fragment",
            RowStage::Fragment,
            Projection::Derived,
            &device,
            None,
        ) {
            RowCompile::Compiled(pipeline) => match *pipeline {
                RowPipeline::Render { pipeline, .. } => pipeline,
                other => panic!("compiled to {other:?}"),
            },
            other => panic!("{other:?}"),
        };
        let extended = VkFeature::ExtendedDynamicState;
        assert!(pipeline.requirements.features.contains(&extended));
        // lavapipe as a context's API-1.2 query reads it: the feature through its extension.
        let mut at_1_2 = description("lavapipe");
        at_1_2.api_version = ApiVersion::V1_2;
        let extensions = at_1_2.extensions.clone();
        at_1_2
            .features
            .retain(|&feature| feature.exposure().exposed(ApiVersion::V1_2, &extensions));
        assert!(at_1_2.has(extended));
        let with = at_1_2.recommended_enables(&DevicePolicy::default());
        assert!(with.features.contains(&extended));
        let mut without = with.clone();
        without.features.remove(&extended);
        let name = format!("{:?}", metal2vulkan::error::Capability::Feature(extended));
        let unmet = unenabled(
            &opened_description(at_1_2.clone(), &without),
            &pipeline.requirements,
        );
        assert!(unmet.contains(&name), "{unmet:?}");
        let unmet = unenabled(&opened_description(at_1_2, &with), &pipeline.requirements);
        assert!(!unmet.contains(&name), "{unmet:?}");
    }

    /// The description of committed profile `stem`, as the Khronos JSON reader builds it.
    fn description(stem: &str) -> DeviceDescription {
        let path = crate::engine::profiles_dir().join(format!("{stem}.json"));
        let text = std::fs::read_to_string(&path).unwrap();
        DeviceDescription::from_khronos_json(&text, None).unwrap()
    }

    #[test]
    fn device_facts_are_checked_against_the_opened_device_not_the_profile() {
        use metal2vulkan::device::{StageCreateFlag, VkLimit};
        let mut description = description("vk12-floor");
        let push_limit = description.limits.max_push_constants_size;
        let enabled = description.recommended_enables(&DevicePolicy::default());
        let opened = opened_profile(description.clone(), &enabled);
        assert!(opened.is_ok(), "{opened:?}");

        let mut requirements = Requirements::default();
        assert!(unmet_device_facts(&opened, &requirements).is_empty());
        // Within the device's limit: met.
        requirements
            .limits
            .insert(VkLimit::MaxPushConstantsSize, u64::from(push_limit));
        assert!(unmet_device_facts(&opened, &requirements).is_empty());
        // Above it (a profile that overstates the device): unmet, named with both values.
        requirements
            .limits
            .insert(VkLimit::MaxPushConstantsSize, u64::from(push_limit) + 4);
        requirements.denorm_independence = Independence::All;
        description.float_controls.denorm_independence = Independence::None;
        let opened = opened_profile(description.clone(), &enabled);
        let unmet = unmet_device_facts(&opened, &requirements);
        assert_eq!(unmet.len(), 2, "{unmet:?}");
        assert!(
            unmet[0].starts_with("Limit { limit: MaxPushConstantsSize"),
            "{unmet:?}"
        );
        assert!(unmet[1].contains("DenormIndependenceAll"), "{unmet:?}");

        // A stage flag needs a feature the context did not enable.
        let mut flagged = Requirements::default();
        flagged
            .stage_flags
            .insert(StageCreateFlag::RequireFullSubgroups);
        assert!(!enabled
            .features
            .contains(&metal2vulkan::device::VkFeature::ComputeFullSubgroups));
        let opened = opened_profile(description.clone(), &enabled);
        assert_eq!(
            unmet_device_facts(&opened, &flagged),
            ["Feature(ComputeFullSubgroups)"]
        );

        // No derivable profile: a row that needs a device fact is unchecked, never created; a
        // row that needs none is unaffected.
        let failed: Result<DeviceProfile, String> = Err("no profile".into());
        assert_eq!(
            unmet_device_facts(&failed, &requirements),
            ["unchecked device facts: no profile"]
        );
        assert!(unmet_device_facts(&failed, &Requirements::default()).is_empty());
    }

    /// The opened description keeps exactly what was enabled, and refuses (through the product's
    /// `DeviceDescription::with_enabled`) an enabled name the device's query does not report or a
    /// feature enabled without the extension that exposes it, rather than dropping either.
    #[test]
    fn the_opened_description_keeps_the_enabled_subset_and_refuses_an_unclosed_set() {
        use metal2vulkan::device::{ApiVersion, VkExtension, VkFeature};
        let description = description("vk12-floor");
        assert_eq!(description.api_version, ApiVersion::V1_2);
        let mut enabled = EnableSet::new();
        enabled.features.insert(VkFeature::RobustBufferAccess);
        let opened = opened_description(description.clone(), &enabled).unwrap();
        assert_eq!(opened.features, enabled.features);
        assert!(opened.extensions.is_empty());
        assert_eq!(opened.api_version, ApiVersion::V1_2);

        assert!(!description.features.contains(&VkFeature::Maintenance8));
        let mut overstated = enabled.clone();
        overstated.features.insert(VkFeature::Maintenance8);
        overstated.extensions.insert(VkExtension::KhrMaintenance8);
        let error = opened_description(description.clone(), &overstated).unwrap_err();
        assert!(error.contains("Maintenance8"), "{error}");

        // A description that lists a promoted feature and its extension, with the feature
        // enabled but not the extension: not exposed, so refused.
        let mut listing = description;
        listing.features.insert(VkFeature::SubgroupSizeControl);
        listing
            .extensions
            .insert(VkExtension::ExtSubgroupSizeControl);
        let mut unexposed = enabled;
        unexposed.features.insert(VkFeature::SubgroupSizeControl);
        let error = opened_description(listing, &unexposed).unwrap_err();
        assert!(error.contains("without exposing"), "{error}");
    }

    #[test]
    fn row_stages_parse_exactly() {
        assert_eq!(RowStage::parse("Kernel").unwrap(), RowStage::Kernel);
        assert_eq!(RowStage::parse("Vertex").unwrap(), RowStage::Vertex);
        assert_eq!(RowStage::parse("Fragment").unwrap(), RowStage::Fragment);
        assert!(RowStage::parse("kernel").is_err());
    }
}
