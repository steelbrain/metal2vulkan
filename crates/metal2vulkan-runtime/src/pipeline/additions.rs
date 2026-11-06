//! The pipeline fields: `requiredThreadsPerThreadgroup`, binary functions (a stage's
//! `binaryFunctions` and the binary functions added to a made state), and what the runtime checks
//! of them.
//!
//! Measured on an M3 Max (macOS 26.7) and an M4 (macOS 27.0.1), which agree:
//! - **Required threads per threadgroup.** Creation stops, with or without API validation, a
//!   size with an axis of 0, one of more threads than the device's limit, and one of more threads
//!   than the descriptor's `maxTotalThreadsPerThreadgroup`; it fails (an error, not a stop) where
//!   that maximum is above the size's threads. API validation stops a direct, threads or indirect
//!   dispatch, and a tile dispatch, whose threadgroup differs on any axis; without it Metal runs
//!   the size given, and so does the runtime. An indirect command buffer's dispatch is not
//!   checked, and runs the size it was encoded with.
//! - **Binary functions, added.** Adding to a stage of a state made without its support flag
//!   (`supportAddingBinaryFunctions`, or the vertex and fragment forms) is an API validation stop;
//!   without validation Metal adds them. The added state has the base state's handles and one for
//!   each added function; an addition the stage already links keeps its handle, and adding
//!   nothing makes a state alike.
//!
//! Measured on an M3 Max under macOS 27.0.1, each state of a source compiled once and never
//! before (a number baked into it):
//! - **A required size is the state's thread limit.** A compute state's
//!   `maxTotalThreadsPerThreadgroup` is the size's threads: 1 for 1x1x1, 21 for 7x3x1, 128 for
//!   8x8x2, 64 for 1x1x64, 1024 for 32x32x1. A tile state's, with
//!   `threadgroupSizeMatchesTileSize` on or off, is the threads rounded down to a multiple of 4:
//!   0 for 1x1, 2x1 and 3x1, 8 for 3x3, 20 for 7x3, 64 for 8x8. A tile dispatch of such a state's
//!   own size runs no thread where the threads are not a multiple of 4, and the command buffer
//!   completes with no error. With no size, both kinds answer 1024. Metal keeps a state it
//!   compiled and hands its limit to a later state of the same function that differs only in the
//!   required size, in the same process (compute and tile) and in a later one (tile). A
//!   measurement of one function under several sizes therefore reads the first size's limit
//!   throughout. The runtime answers each state's own size's threads
//!   (`ComputePipelineState::max_total_threads_per_threadgroup`), and a tile state's dispatch of
//!   a size that is not a multiple of 4 runs (docs/API.md, accepted differences).
//!
//! Measured on an M3 Max under macOS 27.0.1 (`binary_function_linking.m`, of libraries compiled
//! from source and of metallibs alike, plain and with `MTL_DEBUG_LAYER=1`):
//! - **A binary function is a table's function alone.** No visible-function reference resolves to
//!   a function only `binaryFunctions` lists: a direct call to it from a compute, vertex, fragment
//!   or tile function, or from a function `functions` links, fails the creation, plain and under
//!   validation (code 3, "error: unresolved visible function reference: add_one", then "Reason:
//!   visible function not loaded"), which the runtime answers as any reference no linked function
//!   resolves (`Compile`, `NeedsInput(LinkedFunction)`); through a visible-function table, or
//!   with no call, the pipeline is made and its table calls it.
//! - **One name in two lists.** `functions` and `binaryFunctions` holding a function of one name
//!   (one function in both, or two functions of two libraries) is made, plain; API validation
//!   stops it ("All linked functions must have unique names. Function 'add_one' is a duplicate
//!   symbol in binaryFunctions, symbol first found in functions."). A reference calls the one of
//!   `functions`, and a table holds each, each handle calling its own function. Two functions of
//!   one name in `binaryFunctions` are made alike (a table holds both), and a function added to a
//!   made state beside a function of its name the stage links is added, with a handle of its own,
//!   validation or not. `functions` holding one name twice fails the creation (code 3, "error:
//!   duplicated symbol: add_one"), the runtime's `Invalid(DuplicateLinkedFunction)`.
//! - **Not compiled to a binary.** A function made without `MTLFunctionOptionCompileToBinary` in a
//!   stage's `binaryFunctions` fails a plain creation once its stage links, whether a table or
//!   nothing calls it, and when `functions` lists it too: code 2, "Internal error during compute
//!   pipeline creation" (a tile pipeline's "tile pipeline", and an empty description for a render
//!   pipeline's vertex or fragment stage and a mesh pipeline's fragment stage), after a reference
//!   the stage leaves unresolved (code 3 first). Among the functions added to a made state, of its
//!   kernel, vertex, fragment or tile stage, with or without its support flag: code 1, "Failed to
//!   allocate a new pipeline". Added to a stage the state lacks (a drawing state's tile stage, or
//!   the fragment stage it has no function of; a tile state's vertex stage), it is added, as such
//!   functions are, with no handle. API validation stops each ("Function 'add_one' was found in
//!   binaryFunctions array but is not precompiled.", "... in Vertex functions array ..."), and
//!   the runtime answers plain Metal's error ([`Error::MetalError`]), after the compile (a
//!   fragment stage a render pipeline without rasterization drops is not asked: Metal's answer
//!   there was not measured). While a
//!   function `newFunctionWithDescriptor:` made compiled to a binary lives, Metal's
//!   `newFunctionWithName:` of the same library function answers that one object, compiled to a
//!   binary (and the other way about), where the runtime's two functions stay two.
//! - **Mesh pipelines.** A binary function in a mesh pipeline's object or mesh stage ends plain
//!   Metal's process at the creation ("Binary linked functions are not supported for object and
//!   mesh shaders"); its fragment stage takes them as a render pipeline's does.

use super::indirect_arguments::DOMAIN;
use super::render::{GraphicsState, Kind, RenderPipelineState, StateDescriptor};
use super::tile::TileState;
use super::{ComputePipelineState, Kernel, PipelineOptions};
use crate::error::{invalid, Error, MetalError, Problem};
use crate::library::Function;
use crate::values::Size;
use crate::Device;
use metal2vulkan::pipeline::{
    LinkedFunction, MeshPipelineDescriptor, RenderPipelineDescriptor, Stage, StageSource,
};

/// A `requiredThreadsPerThreadgroup` Metal stops at a pipeline's creation.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RequiredThreadsProblem {
    /// A size with an axis of 0, though not every axis (all three 0 is no requirement): "must
    /// not be 0", at creation, with or without API validation.
    ///
    /// Basis: metal-documented (Metal's creation fails "must not be 0", with or without API
    /// validation).
    EmptyAxis {
        /// The size given.
        required: [u32; 3],
    },
    /// A size of more threads than the device's limit (its `maxTotalThreadsPerThreadgroup`
    /// answer): "must be <= 1024 (device limit)", at creation, with or without API validation.
    ///
    /// Basis: metal-documented (Metal's creation fails "must be <= 1024 (device limit)", with or
    /// without API validation).
    OverDeviceLimit {
        /// The size's threads.
        threads: u64,
        /// The device's limit.
        limit: u32,
    },
    /// A size of more threads than the descriptor's `maxTotalThreadsPerThreadgroup`: "must be <=
    /// maxTotalThreadsPerThreadgroup (32)", at creation, with or without API validation.
    ///
    /// Basis: metal-documented (Metal's creation fails "must be <= maxTotalThreadsPerThreadgroup
    /// (32)", with or without API validation).
    OverMaxTotal {
        /// The size's threads.
        threads: u64,
        /// The descriptor's `maxTotalThreadsPerThreadgroup`.
        max: u32,
    },
    /// A descriptor's `maxTotalThreadsPerThreadgroup` above the size's threads: Metal's creation
    /// fails, with or without API validation ("Kernel specified max total threads per
    /// threadgroup (64) must match compute pipeline specified max total threads per threadgroup
    /// (256)").
    ///
    /// Basis: metal-documented (Metal's creation fails "Kernel specified max total threads per
    /// threadgroup (64) must match compute pipeline specified max total threads per threadgroup
    /// (256)", with or without API validation).
    MaxTotalDiffers {
        /// The size's threads.
        threads: u64,
        /// The descriptor's `maxTotalThreadsPerThreadgroup`.
        max: u32,
    },
}

/// The stage of a state that binary functions are added to.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinaryFunctionStage {
    /// A compute state's kernel.
    Kernel,
    /// A render state's vertex function.
    Vertex,
    /// A render state's fragment function.
    Fragment,
    /// A tile pipeline's tile function.
    Tile,
}

/// Binary functions Metal stops.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BinaryFunctionProblem {
    /// Functions added to any stage of a mesh state, whose descriptor has no support flag:
    /// Metal's API validation stops functions added to a stage without its flag ("Pipeline
    /// state is adding fragment functions but it was created with
    /// supportAddingFragmentBinaryFunctions=NO"), and plain Metal's process ends at the call for
    /// a mesh state, whether the functions are added to its fragment stage or to the vertex or
    /// tile stage it lacks (an M4 under macOS 27.0.1: `mesh_frag_two`,
    /// `mesh_frag_one_add_two`, `mesh_vertex_add`, `mesh_tile_add`). Plain Metal's answer is
    /// the runtime's for a compute state's kernel and a drawing or tile state's own stages made
    /// without their flags, and for a stage a drawing or tile state lacks.
    ///
    /// Basis: metal-measured (an M4, macOS 27.0.1: plain Metal's process ends at the call for a
    /// mesh state — cases `mesh_frag_two`, `mesh_frag_one_add_two`, `mesh_vertex_add` and
    /// `mesh_tile_add`).
    NotSupported {
        /// The stage.
        stage: BinaryFunctionStage,
    },
    /// A function not compiled to a binary in a stage's `binary_functions` or among the functions
    /// added to a stage of a made state, which plain Metal's creation fails over: the runtime
    /// answers it as [`Error::MetalError`], code 2 for a stage's ("Internal error during compute
    /// pipeline creation", "Internal error during tile pipeline creation", or an empty
    /// description for a render or mesh pipeline's stage) once its stage compiled, and code 1
    /// ("Failed to allocate a new pipeline") for one added. Metal's API validation stops both
    /// ("Function 'f' was found in binaryFunctions array but is not precompiled.", "Function 'f'
    /// was found in Fragment functions array but is not precompiled."); a function added to a
    /// stage the state lacks is taken, as plain Metal takes it (the module docs).
    ///
    /// Basis: metal-measured (an M3 Max, macOS 27.0.1: plain Metal's creation fails with code 2,
    /// an addition's with code 1, `binary_function_linking.m`).
    NotPrecompiled {
        /// The function's name.
        name: String,
    },
    /// Binary functions in a mesh render pipeline's object or mesh stage (the `binary_functions`
    /// of its function's linked functions, Metal's `objectLinkedFunctions` and
    /// `meshLinkedFunctions`): plain Metal's process ends at the creation, at its descriptor
    /// validation ("Binary linked functions are not supported for object and mesh shaders"). Its
    /// fragment stage takes them.
    ///
    /// Basis: metal-measured (an M3 Max, macOS 27.0.1: plain Metal's process ends at the creation,
    /// `binary_function_linking.m`).
    ObjectOrMeshStage {
        /// The stage: `Object` or `Mesh`.
        stage: Stage,
    },
    /// Functions added to a stage the compile layer already compiled (`StageSource::Compiled`),
    /// whose linked functions are fixed: Metal's stages are functions, so it has no such call.
    ///
    /// Basis: ours (this crate's already-compiled stage: Metal's descriptor takes a function, so it
    /// has no such call).
    CompiledStage {
        /// The stage.
        stage: BinaryFunctionStage,
    },
}

/// `MTLRenderPipelineFunctionsDescriptor`: the binary functions
/// [`RenderPipelineState::new_render_pipeline_state_with_additional_binary_functions`] adds to
/// each stage.
#[non_exhaustive]
#[derive(Clone, Debug, Default)]
pub struct RenderPipelineFunctionsDescriptor {
    /// `vertexAdditionalBinaryFunctions`.
    pub vertex_additional_binary_functions: Vec<Function>,
    /// `fragmentAdditionalBinaryFunctions`.
    pub fragment_additional_binary_functions: Vec<Function>,
    /// `tileAdditionalBinaryFunctions`.
    pub tile_additional_binary_functions: Vec<Function>,
}

impl RenderPipelineFunctionsDescriptor {
    /// A descriptor that adds nothing.
    pub fn new() -> RenderPipelineFunctionsDescriptor {
        RenderPipelineFunctionsDescriptor::default()
    }
}

/// `required`, with Metal's all-zero size as none.
pub(super) fn required_size(required: Option<[u32; 3]>) -> Option<[u32; 3]> {
    required.filter(|axes| *axes != [0; 3])
}

impl Kernel {
    /// The descriptor's `requiredThreadsPerThreadgroup` and `maxTotalThreadsPerThreadgroup`.
    fn threadgroup_fields(&self) -> (Option<[u32; 3]>, Option<u32>) {
        match self {
            Kernel::Compute(d) => (
                d.required_threads_per_threadgroup,
                d.max_total_threads_per_threadgroup,
            ),
            Kernel::Tile(d) => (
                d.required_threads_per_threadgroup,
                d.max_total_threads_per_threadgroup,
            ),
        }
    }
}

/// Check, before compiling, a compute or tile creation of `kernel` in `call` on `device`: its
/// required threads per threadgroup against the device's limit and its own maximum, as Metal does
/// (the module docs).
pub(super) fn check_kernel(
    device: &Device,
    kernel: &Kernel,
    call: &'static str,
) -> Result<(), Error> {
    let (required, max) = kernel.threadgroup_fields();
    if let Some(required) = required_size(required) {
        let problem = |p| invalid(call, Problem::RequiredThreads(p));
        if required.contains(&0) {
            return Err(problem(RequiredThreadsProblem::EmptyAxis { required }));
        }
        let threads: u64 = required.iter().map(|&axis| u64::from(axis)).product();
        let limit = device
            .compile_device()
            .metal_properties()
            .max_total_threads_per_threadgroup;
        if threads > u64::from(limit) {
            return Err(problem(RequiredThreadsProblem::OverDeviceLimit {
                threads,
                limit,
            }));
        }
        match max.filter(|&max| max > 0) {
            Some(max) if threads > u64::from(max) => {
                return Err(problem(RequiredThreadsProblem::OverMaxTotal {
                    threads,
                    max,
                }));
            }
            Some(max) if u64::from(max) > threads => {
                return Err(problem(RequiredThreadsProblem::MaxTotalDiffers {
                    threads,
                    max,
                }));
            }
            _ => {}
        }
    }
    Ok(())
}

/// The kind of pipeline a creation makes, whose name Metal's error for a binary function not
/// compiled to a binary says.
#[derive(Clone, Copy, Debug)]
pub(super) enum Made {
    Compute,
    Tile,
    /// A render pipeline's vertex or fragment stage, or a mesh pipeline's fragment stage.
    Render,
}

impl Made {
    /// Metal's `localizedDescription` for such a function of a stage of the pipeline.
    fn description(self) -> &'static str {
        match self {
            Made::Compute => "Internal error during compute pipeline creation",
            Made::Tile => "Internal error during tile pipeline creation",
            Made::Render => "",
        }
    }
}

/// The code of Metal's error for a stage's binary function not compiled to a binary.
const NOT_PRECOMPILED: i64 = 2;

/// The code and the `localizedDescription` of Metal's error for a function added to a made state
/// that is not compiled to a binary.
const ADDED_NOT_PRECOMPILED: (i64, &str) = (1, "Failed to allocate a new pipeline");

/// Metal's error in `call` for `f`, not compiled to a binary, of `code` and `description`.
fn not_precompiled(
    call: &'static str,
    f: &LinkedFunction,
    (code, description): (i64, &'static str),
) -> Error {
    Error::MetalError(MetalError::new(
        call,
        Problem::BinaryFunctions(BinaryFunctionProblem::NotPrecompiled {
            name: f.name().to_owned(),
        }),
        DOMAIN,
        code,
        description,
    ))
}

/// Refused in `call` once the stages compiled, as plain Metal fails a creation of kind `made`
/// whose stage links a binary function not compiled to a binary (the module docs): the first such
/// function of `stages`, in order. A compiled stage links none Metal names.
pub(super) fn check_precompiled<'a>(
    call: &'static str,
    made: Made,
    stages: impl IntoIterator<Item = &'a StageSource>,
) -> Result<(), Error> {
    for stage in stages {
        let StageSource::Function(function) = stage else {
            continue;
        };
        if let Some(f) = function
            .linked
            .binary_functions
            .iter()
            .find(|f| !f.precompiled)
        {
            return Err(not_precompiled(
                call,
                f,
                (NOT_PRECOMPILED, made.description()),
            ));
        }
    }
    Ok(())
}

/// Refused in `call` before anything, as plain Metal's descriptor validation ends its process
/// there: a mesh pipeline's object or mesh stage that links binary functions (the module docs).
pub(super) fn check_mesh(
    call: &'static str,
    descriptor: &MeshPipelineDescriptor,
) -> Result<(), Error> {
    let stages = descriptor
        .object
        .iter()
        .map(|object| (Stage::Object, object))
        .chain([(Stage::Mesh, &descriptor.mesh)]);
    for (stage, source) in stages {
        if let StageSource::Function(function) = source {
            if !function.linked.binary_functions.is_empty() {
                return Err(invalid(
                    call,
                    Problem::BinaryFunctions(BinaryFunctionProblem::ObjectOrMeshStage { stage }),
                ));
            }
        }
    }
    Ok(())
}

/// [`check_precompiled`] of a render creation's vertex and fragment stages.
pub(super) fn check_render(
    call: &'static str,
    descriptor: &RenderPipelineDescriptor,
) -> Result<(), Error> {
    check_precompiled(
        call,
        Made::Render,
        std::iter::once(&descriptor.vertex).chain(&descriptor.fragment),
    )
}

/// `stage` of a state, with `functions` added to its `binary_functions`, in `call` on `device`;
/// `supported` is the stage's support flag. Each function must be of `device` and precompiled
/// (Metal's error otherwise, the module docs); one the stage's `functions` or `binary_functions`
/// already link (the same name it links under, library function, AIR, constant values and
/// specialization) keeps its entry and handle, and another of a name the stage links is added
/// beside it.
fn added(
    device: &Device,
    call: &'static str,
    stage: (&StageSource, BinaryFunctionStage),
    supported: bool,
    functions: &[Function],
) -> Result<StageSource, Error> {
    let (source, which) = stage;
    if functions.is_empty() {
        return Ok(source.clone());
    }
    if !supported {
        return Err(invalid(
            call,
            Problem::BinaryFunctions(BinaryFunctionProblem::NotSupported { stage: which }),
        ));
    }
    let StageSource::Function(descriptor) = source else {
        return Err(invalid(
            call,
            Problem::BinaryFunctions(BinaryFunctionProblem::CompiledStage { stage: which }),
        ));
    };
    let mut descriptor = descriptor.clone();
    for function in functions {
        if !function.device().same(device) {
            return Err(invalid(call, Problem::ForeignObject));
        }
        let linked = function.linked_function();
        if !linked.precompiled {
            return Err(not_precompiled(call, &linked, ADDED_NOT_PRECOMPILED));
        }
        let same = |f: &LinkedFunction| {
            f.name() == linked.name()
                && f.function.info().name == linked.function.info().name
                && f.function.info().air_hash == linked.function.info().air_hash
                && f.constants == linked.constants
                && f.specialized == linked.specialized
        };
        let held = descriptor.linked.functions.iter().any(same)
            || descriptor.linked.binary_functions.iter().any(same);
        if !held {
            descriptor.linked.binary_functions.push(linked);
        }
    }
    Ok(StageSource::Function(descriptor))
}

/// A stage's support flag, for a stage the state lacks.
const ABSENT: bool = false;

/// How many `binary_functions` `stage`'s function links: none for a compiled stage.
fn binary_count(stage: &StageSource) -> usize {
    match stage {
        StageSource::Function(descriptor) => descriptor.linked.binary_functions.len(),
        _ => 0,
    }
}

/// [`added`] of `stage`, whose support flag is `supported`, linking `functions` as with the flag,
/// and how many functions it newly links without the flag: none where it is set. Plain Metal
/// makes the state without the flag, where its API validation stops it, and its handle of such
/// a function calls the stage's last other callable function, which a table of the state takes it
/// as (`function_table.rs`).
fn added_counting(
    device: &Device,
    call: &'static str,
    stage: (&StageSource, BinaryFunctionStage),
    supported: bool,
    functions: &[Function],
) -> Result<(StageSource, usize), Error> {
    let before = binary_count(stage.0);
    let source = added(device, call, stage, true, functions)?;
    let unsupported = match supported {
        true => 0,
        false => binary_count(&source) - before,
    };
    Ok((source, unsupported))
}

/// `functions` added in `call` to a stage a drawing or tile state lacks, which plain Metal makes
/// the state without, giving no handle of them, where its API validation stops them, a function
/// not compiled to a binary among them (the module docs): refused (`Invalid`) only for a function
/// of another device.
fn ignored(device: &Device, call: &'static str, functions: &[Function]) -> Result<(), Error> {
    for function in functions {
        if !function.device().same(device) {
            return Err(invalid(call, Problem::ForeignObject));
        }
    }
    Ok(())
}

impl ComputePipelineState {
    /// `newComputePipelineStateWithAdditionalBinaryFunctions:error:`: a new state of this one's
    /// descriptor with `functions` added to its kernel's `binary_functions`, made as
    /// [`Device::new_compute_pipeline_state`] makes one with the default options and this state's
    /// label. It has this state's function handles and one for each added function; adding nothing
    /// makes a state alike. On a state made without `support_adding_binary_functions`, which
    /// Metal's API validation stops, an added function's handle calls the kernel's last other
    /// callable function, and none where it has none (an entry that reads null), as plain Metal's
    /// does. A function not made compiled to a binary (`FunctionOptions::COMPILE_TO_BINARY`) fails
    /// it as plain Metal's call fails, with or without the support flag: [`Error::MetalError`]
    /// with `BinaryFunctions(NotPrecompiled)`, code 1 ("Failed to allocate a new pipeline"), where
    /// Metal's API validation stops it. Refused (`Invalid`): a function of another device
    /// (`ForeignObject`). Otherwise refused as the creation is.
    pub fn new_compute_pipeline_state_with_additional_binary_functions(
        &self,
        functions: &[Function],
    ) -> Result<ComputePipelineState, Error> {
        const CALL: &str = "new_compute_pipeline_state_with_additional_binary_functions";
        let device = self.device().clone();
        let Kernel::Compute(descriptor) = &self.0.descriptor else {
            return Err(invalid(
                CALL,
                Problem::BinaryFunctions(BinaryFunctionProblem::NotSupported {
                    stage: BinaryFunctionStage::Kernel,
                }),
            ));
        };
        let mut descriptor = descriptor.clone();
        let unsupported;
        (descriptor.kernel, unsupported) = added_counting(
            &device,
            CALL,
            (&descriptor.kernel, BinaryFunctionStage::Kernel),
            descriptor.support_adding_binary_functions,
            functions,
        )?;
        ComputePipelineState::create_adding(
            &device,
            &descriptor,
            &labelled(self.label()),
            None,
            self.added_unsupported() + unsupported,
        )
    }

    /// `requiredThreadsPerThreadgroup`: the descriptor's, a zero size where it requires none.
    pub fn required_threads_per_threadgroup(&self) -> Size {
        size(self.required_threads())
    }

    /// The threadgroup size every dispatch of the state must have, if the descriptor requires
    /// one.
    pub(crate) fn required_threads(&self) -> Option<[u32; 3]> {
        required_size(self.0.descriptor.threadgroup_fields().0)
    }
}

/// The default options with `label`: a state made with additional binary functions keeps its
/// original's label, as Metal's does.
fn labelled(label: Option<&str>) -> PipelineOptions {
    PipelineOptions {
        label: label.map(str::to_owned),
        ..PipelineOptions::default()
    }
}

/// `required` as Metal's `MTLSize`, zero where none.
fn size(required: Option<[u32; 3]>) -> Size {
    let [width, height, depth] = required.unwrap_or_default().map(u64::from);
    Size::new(width, height, depth)
}

/// `new_render_pipeline_state_with_additional_binary_functions` of a mesh render pipeline's state
/// of `descriptor`, in `call`: its descriptor supports adding binary functions to none of the
/// stages the call names, so any function listed is refused as one added to a stage without its
/// support flag, where plain Metal's process ends; adding none makes a state alike, with the
/// default options and the original's `label`.
fn mesh_with_binary_functions(
    device: &Device,
    call: &'static str,
    (descriptor, label): (&MeshPipelineDescriptor, Option<&str>),
    functions: &RenderPipelineFunctionsDescriptor,
) -> Result<RenderPipelineState, Error> {
    for (list, stage) in [
        (
            &functions.vertex_additional_binary_functions,
            BinaryFunctionStage::Vertex,
        ),
        (
            &functions.fragment_additional_binary_functions,
            BinaryFunctionStage::Fragment,
        ),
        (
            &functions.tile_additional_binary_functions,
            BinaryFunctionStage::Tile,
        ),
    ] {
        added(device, call, (&descriptor.mesh, stage), ABSENT, list)?;
    }
    GraphicsState::create_mesh(device, descriptor, &labelled(label))
        .map(RenderPipelineState::of_graphics)
}

impl RenderPipelineState {
    /// `newRenderPipelineStateWithAdditionalBinaryFunctions:error:`: a new state of this one's
    /// descriptor with `functions`' lists added to its stages' `binary_functions`, made as
    /// [`Device::new_render_pipeline_state`] or [`Device::new_tile_render_pipeline_state`] makes
    /// one with the default options and this state's label. It has this state's function handles
    /// and one for each added function. Functions added to a stage the state lacks (a drawing
    /// state's tile stage or missing fragment stage, a tile state's vertex and fragment stages) are
    /// left out, with no handle, compiled to a binary or not, as plain Metal leaves them where its
    /// API validation stops them; on
    /// a stage made without its support flag (`support_adding_vertex_binary_functions`,
    /// `support_adding_fragment_binary_functions`, a tile descriptor's
    /// `support_adding_binary_functions`), which its API validation also stops, an added function's
    /// handle calls the stage's last other callable function, and none where it has none (an entry
    /// that reads null), as plain Metal's does. A function not made compiled to a binary added to
    /// a stage the state has fails it as plain Metal's call fails: [`Error::MetalError`] with
    /// `BinaryFunctions(NotPrecompiled)`, code 1 ("Failed to allocate a new pipeline"), where
    /// Metal's API validation stops it. Refused (`Invalid`): functions added to a mesh state
    /// (`BinaryFunctions(NotSupported)`; plain Metal's process ends there); a function of another
    /// device (`ForeignObject`); and functions added to a stage the compile layer already compiled
    /// (`BinaryFunctions(CompiledStage)`). Otherwise refused as the creation is.
    pub fn new_render_pipeline_state_with_additional_binary_functions(
        &self,
        functions: &RenderPipelineFunctionsDescriptor,
    ) -> Result<RenderPipelineState, Error> {
        const CALL: &str = "new_render_pipeline_state_with_additional_binary_functions";
        let device = self.device().clone();
        match self.kind() {
            Kind::Graphics(state) => {
                let descriptor = match state.descriptor() {
                    StateDescriptor::Vertex(descriptor) => descriptor,
                    StateDescriptor::Mesh(descriptor) => {
                        return mesh_with_binary_functions(
                            &device,
                            CALL,
                            (descriptor, self.label()),
                            functions,
                        )
                    }
                };
                ignored(&device, CALL, &functions.tile_additional_binary_functions)?;
                let mut descriptor = descriptor.clone();
                let vertex;
                (descriptor.vertex, vertex) = added_counting(
                    &device,
                    CALL,
                    (&descriptor.vertex, BinaryFunctionStage::Vertex),
                    descriptor.support_adding_vertex_binary_functions,
                    &functions.vertex_additional_binary_functions,
                )?;
                let mut fragment = 0;
                if let Some(source) = &descriptor.fragment {
                    let (added, unsupported) = added_counting(
                        &device,
                        CALL,
                        (source, BinaryFunctionStage::Fragment),
                        descriptor.support_adding_fragment_binary_functions,
                        &functions.fragment_additional_binary_functions,
                    )?;
                    descriptor.fragment = Some(added);
                    fragment = unsupported;
                } else {
                    ignored(
                        &device,
                        CALL,
                        &functions.fragment_additional_binary_functions,
                    )?;
                }
                GraphicsState::create_adding(
                    &device,
                    &descriptor,
                    &labelled(self.label()),
                    None,
                    [
                        state.added_unsupported(Stage::Vertex) + vertex,
                        state.added_unsupported(Stage::Fragment) + fragment,
                    ],
                )
                .map(RenderPipelineState::of_graphics)
            }
            Kind::Tile(state) => {
                let mut descriptor = state.descriptor().clone();
                for list in [
                    &functions.vertex_additional_binary_functions,
                    &functions.fragment_additional_binary_functions,
                ] {
                    ignored(&device, CALL, list)?;
                }
                let unsupported;
                (descriptor.tile, unsupported) = added_counting(
                    &device,
                    CALL,
                    (&descriptor.tile, BinaryFunctionStage::Tile),
                    descriptor.support_adding_binary_functions,
                    &functions.tile_additional_binary_functions,
                )?;
                TileState::create_adding(
                    &device,
                    &descriptor,
                    &labelled(self.label()),
                    None,
                    state.kernel().added_unsupported() + unsupported,
                )
                .map(RenderPipelineState::of_tile)
            }
        }
    }

    /// `requiredThreadsPerTileThreadgroup`: a tile pipeline's descriptor's
    /// `required_threads_per_threadgroup`, a zero size where it requires none and for a drawing
    /// state.
    pub fn required_threads_per_tile_threadgroup(&self) -> Size {
        match self.kind() {
            Kind::Graphics(_) => size(None),
            Kind::Tile(state) => size(state.kernel().required_threads()),
        }
    }

    /// `requiredThreadsPerObjectThreadgroup`: a mesh state's descriptor's
    /// `required_threads_per_object_threadgroup`, a zero size where it requires none and for any
    /// other state, as an M3 Max answers the descriptor's. The descriptor's that Metal refuses
    /// are refused at the creation ([`Device::new_mesh_render_pipeline_state`]); a draw is not
    /// held to it.
    pub fn required_threads_per_object_threadgroup(&self) -> Size {
        size(
            self.mesh_descriptor()
                .and_then(|d| required_size(d.required_threads_per_object_threadgroup)),
        )
    }

    /// `requiredThreadsPerMeshThreadgroup`: as
    /// [`required_threads_per_object_threadgroup`](Self::required_threads_per_object_threadgroup),
    /// of the mesh stage.
    pub fn required_threads_per_mesh_threadgroup(&self) -> Size {
        size(
            self.mesh_descriptor()
                .and_then(|d| required_size(d.required_threads_per_mesh_threadgroup)),
        )
    }

    /// The descriptor of a mesh state; `None` for any other state.
    fn mesh_descriptor(&self) -> Option<&MeshPipelineDescriptor> {
        match self.kind() {
            Kind::Graphics(state) => match state.descriptor() {
                StateDescriptor::Mesh(d) => Some(d),
                StateDescriptor::Vertex(_) => None,
            },
            Kind::Tile(_) => None,
        }
    }
}
